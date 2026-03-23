"""
XSTest exploration with Qwen/Qwen3Guard-Gen-4B.

Two modes:
  1. Inference  — runs the model, saves embeddings + metadata to EMBEDDINGS_DIR
  2. Plot only  — loads saved data, rebuilds PCA/UMAP plots without touching the model

The script auto-detects which mode to use:
  - if embeddings.npy + meta.parquet exist in EMBEDDINGS_DIR → skip inference
  - otherwise → run inference and save
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import Counter
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from umap import UMAP

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3Guard-Gen-4B"
DEVICE = "mps"
EMBEDDINGS_DIR = "data/xstest_qwen3guard_4b"
PLOTS_DIR = "pca_presentation/pca_umap_layers_xstest_qwen_4"
MAX_NEW_TOKENS = 64
HF_TOKEN = os.environ.get("HF_TOKEN", "")

EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
META_PATH = os.path.join(EMBEDDINGS_DIR, "meta.parquet")
PROJECTIONS_DIR = os.path.join(EMBEDDINGS_DIR, "projections")
SILHOUETTE_PATH = os.path.join(EMBEDDINGS_DIR, "silhouette_scores.npy")

# ── Load or run inference ─────────────────────────────────────────────────────
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH):
    print(f"Found saved embeddings in {EMBEDDINGS_DIR}, skipping inference.")
    embeddings = np.load(EMBEDDINGS_PATH)          # (n_samples, n_layers, hidden_dim)
    meta = pd.read_parquet(META_PATH)
else:
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        login()

    # Dataset
    ds = load_dataset("walledai/XSTest")
    df = pd.DataFrame(ds["test"])
    df["focus"] = df["focus"].fillna("unknown")
    print(f"Dataset size: {len(df)}, label distribution:\n{df['label'].value_counts()}")

    # Model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    model.eval()

    def extract_label_and_categories(content):
        safe_match = re.search(r"Safety: (Safe|Unsafe|Controversial)", content)
        label = safe_match.group(1) if safe_match else None
        categories = re.findall(
            r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
            r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
            r"Copyright Violation|Jailbreak|None)",
            content,
        )
        return label, categories

    def run_guardrail(prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        new_ids = generated_ids[0][input_len:]
        content = tokenizer.decode(new_ids, skip_special_tokens=True)
        label_str, _ = extract_label_and_categories(content)
        pred = 1 if label_str == "Unsafe" else 0

        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        all_hidden = np.stack([
            fwd.hidden_states[i][0, -1, :].cpu().float().numpy()
            for i in range(len(fwd.hidden_states))
        ])

        return pred, label_str, all_hidden

    # Inference
    rows = []
    all_embeddings = []

    for _, sample in tqdm(df.iterrows(), total=len(df)):
        true_label = 1 if sample["label"] == "unsafe" else 0
        pred, label_str, hidden = run_guardrail(sample["prompt"])

        if pred == 1 and true_label == 1:
            category = "TP"
        elif pred == 0 and true_label == 0:
            category = "TN"
        elif pred == 1 and true_label == 0:
            category = "FP"
        else:
            category = "FN"

        rows.append({
            "text": sample["prompt"],
            "category": category,
            "true_label": true_label,
            "pred": pred,
            "type": sample["type"],
            "focus": sample["focus"],
        })
        all_embeddings.append(hidden)

    embeddings = np.stack(all_embeddings)   # (n_samples, n_layers, hidden_dim)
    meta = pd.DataFrame(rows)

    # Save
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    meta.to_parquet(META_PATH, index=False)
    print(f"Saved embeddings {embeddings.shape} → {EMBEDDINGS_PATH}")
    print(f"Saved metadata   {meta.shape}       → {META_PATH}")

print(Counter(meta["category"]))

# ── Visualization ─────────────────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)

n_layers = embeddings.shape[1]
quad_labels = meta["category"].to_numpy()
type_labels = meta["type"].to_numpy()

quad_colors = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e67e22", "FN": "#e74c3c"}


def make_color_map(labels_arr):
    unique = sorted(set(labels_arr))
    cmap = plt.get_cmap("tab20", len(unique))
    return {lbl: cmap(i) for i, lbl in enumerate(unique)}


def scatter_panel(ax, X_2d, labels_arr, color_map, title, sil=None):
    for lbl, color in color_map.items():
        mask = labels_arr == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[color], label=f"{lbl} ({mask.sum()})",
                   alpha=0.7, s=20, edgecolors="none")
    title_str = title if sil is None else f"{title}\nSilhouette: {sil:.3f}"
    ax.set_title(title_str, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", markerscale=1.2)


os.makedirs(PROJECTIONS_DIR, exist_ok=True)

# Load cached silhouette scores if all projections already exist
_all_cached = all(
    os.path.exists(os.path.join(PROJECTIONS_DIR, f"layer_{i:02d}_pca.npy")) and
    os.path.exists(os.path.join(PROJECTIONS_DIR, f"layer_{i:02d}_umap.npy"))
    for i in range(n_layers)
)
if _all_cached and os.path.exists(SILHOUETTE_PATH):
    silhouette_scores = np.load(SILHOUETTE_PATH).tolist()
    print(f"Found cached projections in {PROJECTIONS_DIR}, skipping PCA/UMAP computation.")
else:
    silhouette_scores = []

type_cmap = make_color_map(type_labels)

for layer_idx in tqdm(range(n_layers), desc="Plotting layers"):
    pca_path  = os.path.join(PROJECTIONS_DIR, f"layer_{layer_idx:02d}_pca.npy")
    umap_path = os.path.join(PROJECTIONS_DIR, f"layer_{layer_idx:02d}_umap.npy")

    if os.path.exists(pca_path) and os.path.exists(umap_path):
        X_pca  = np.load(pca_path)
        X_umap = np.load(umap_path)
        sil = silhouette_scores[layer_idx] if len(silhouette_scores) > layer_idx else (
            silhouette_score(X_pca, quad_labels) if len(set(quad_labels)) > 1 else 0.0
        )
    else:
        X = embeddings[:, layer_idx, :]      # (n_samples, hidden_dim)
        X_pca  = PCA(n_components=2).fit_transform(X)
        X_umap = UMAP(n_components=2, random_state=42, verbose=False).fit_transform(X)
        np.save(pca_path,  X_pca)
        np.save(umap_path, X_umap)
        sil = silhouette_score(X_pca, quad_labels) if len(set(quad_labels)) > 1 else 0.0
        silhouette_scores.append(sil)
        np.save(SILHOUETTE_PATH, np.array(silhouette_scores))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Layer {layer_idx} | Qwen3Guard-Gen-4B | XSTest", fontsize=11)

    scatter_panel(axes[0, 0], X_pca,  quad_labels, quad_colors, "PCA  — TP/TN/FP/FN", sil)
    scatter_panel(axes[0, 1], X_umap, quad_labels, quad_colors, "UMAP — TP/TN/FP/FN")
    scatter_panel(axes[1, 0], X_pca,  type_labels, type_cmap,   "PCA  — type")
    scatter_panel(axes[1, 1], X_umap, type_labels, type_cmap,   "UMAP — type")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/layer_{layer_idx:02d}.png", dpi=120, bbox_inches="tight")
    plt.close()

# ── Silhouette summary ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(n_layers), silhouette_scores, marker="o", markersize=4, linewidth=1.5)
ax.axhline(
    y=max(silhouette_scores), color="red", linestyle="--", alpha=0.5,
    label=f"max={max(silhouette_scores):.3f} @ layer {np.argmax(silhouette_scores)}",
)
ax.set_xlabel("Layer")
ax.set_ylabel("Silhouette score")
ax.set_title("Separability of TP/TN/FP/FN by layer (XSTest, Qwen3Guard-Gen-4B)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/silhouette_by_layer.png", dpi=150)
plt.close()

best = int(np.argmax(silhouette_scores))
print(f"Best layer: {best} (silhouette={silhouette_scores[best]:.4f})")
print(f"Plots saved to ./{PLOTS_DIR}/")
