
import sys, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc
import anndata as ad
import scipy.sparse

plt.rcParams['svg.fonttype'] = 'none'

# ── anndata compatibility shim (old pkl → new anndata) ──────────────────────
try:
    import anndata._core.file_backing as _afb
    _orig = _afb.AnnDataFileManager.__setstate__
    def _compat(self, state):
        try:
            _orig(self, state)
        except KeyError as e:
            if "_adata_ref" in str(e):
                for k, v in state.items():
                    self.__dict__[k] = v
                self.__dict__.setdefault("_adata_ref", None)
            else:
                raise
    _afb.AnnDataFileManager.__setstate__ = _compat
except Exception:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ROOT      = HERE.parent
DATA_DIR  = ROOT / "data"
ISTAR_OUT = HERE / "istar_outputs"
OUT_DIR   = HERE           # SVGs saved alongside this script

ALIGNED_VIS = DATA_DIR / "breastcancer_sample1_rep1_aligned_tovisiumimage"
ALIGNED_XEN = DATA_DIR / "breastcancer_sample1_rep1_aligned_toxeniumimage"

# ── Load iStar correlation DataFrames ────────────────────────────────────────
print("Loading iStar correlation results …")
vis_corr = pd.read_csv(ISTAR_OUT / "visium/correlation_df.csv")
xen_corr = pd.read_csv(ISTAR_OUT / "xenium/correlation_df.csv")

# Sort by gene for aligned scatter plots
vis_corr = vis_corr.sort_values("Gene").reset_index(drop=True)
xen_corr = xen_corr.sort_values("Gene").reset_index(drop=True)

print(f"  Visium  — mean Pearson: {vis_corr['Pearson'].mean():.4f}  n_genes: {len(vis_corr)}")
print(f"  Xenium  — mean Pearson: {xen_corr['Pearson'].mean():.4f}  n_genes: {len(xen_corr)}")

# ── Fig 2a — Pearson histogram ───────────────────────────────────────────────
print("\nPlotting Pearson histogram …")
plt.figure(figsize=(10, 5))
sns.histplot(vis_corr["Pearson"], color="#55B4E9", label="iStar — Visium data", kde=True)
sns.histplot(xen_corr["Pearson"], color="#E69F01", label="iStar — Xenium data", kde=True)

vis_mean = vis_corr["Pearson"].mean()
xen_mean = xen_corr["Pearson"].mean()
plt.axvline(vis_mean, color="#55B4E9", linestyle="--")
plt.axvline(xen_mean, color="#E69F01", linestyle="--")
plt.text(vis_mean, plt.ylim()[1], f"{vis_mean:.3f}", fontsize=10, color="#55B4E9", ha="center")
plt.text(xen_mean, plt.ylim()[1], f"{xen_mean:.3f}", fontsize=10, color="#E69F01", ha="center")
plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
plt.xlim(0, 1)
plt.legend()
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/istar_fig2a_histogram.svg", dpi=1000, bbox_inches="tight")
# plt.close()
plt.show()


# ── Fig 2a — rMSE_range histogram ────────────────────────────────────────────
print("Plotting rMSE_range histogram …")
plt.figure(figsize=(10, 5))
sns.histplot(vis_corr["rMSE_range"], color="#55B4E9", label="iStar — Visium data", kde=True)
sns.histplot(xen_corr["rMSE_range"], color="#E69F01", label="iStar — Xenium data", kde=True)

vis_mean_r = vis_corr["rMSE_range"].mean()
xen_mean_r = xen_corr["rMSE_range"].mean()
plt.axvline(vis_mean_r, color="#55B4E9", linestyle="--")
plt.axvline(xen_mean_r, color="#E69F01", linestyle="--")
plt.text(vis_mean_r, plt.ylim()[1], f"{vis_mean_r:.3f}", fontsize=10, color="#55B4E9", ha="center")
plt.text(xen_mean_r, plt.ylim()[1], f"{xen_mean_r:.3f}", fontsize=10, color="#E69F01", ha="center")
plt.xlabel("Normalised rMSE (range)")
plt.ylabel("Frequency")
plt.legend()
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/istar_fig2a_histogram_rmse.svg", dpi=1000, bbox_inches="tight")
# plt.close()
plt.show()



xen_corr
vis_corr



# # ── Align gene lists (inner join) for scatter plots ──────────────────────────
# # iStar may predict slightly different gene sets per modality; keep common genes.
# shared_genes = sorted(set(vis_corr["Gene"]) & set(xen_corr["Gene"]))
# vis_s = vis_corr[vis_corr["Gene"].isin(shared_genes)].sort_values("Gene").reset_index(drop=True)
# xen_s = xen_corr[xen_corr["Gene"].isin(shared_genes)].sort_values("Gene").reset_index(drop=True)
# print(f"\nShared genes for scatter: {len(shared_genes)}")

# genes_highlight = ["HDC", "GZMK", "AHSP", "ANKRD30A"]

# # ── Fig 2b — Pearson scatter ─────────────────────────────────────────────────
# print("Plotting Pearson scatter …")
# plt.figure(figsize=(10, 5))
# plt.scatter(vis_s["Pearson"], xen_s["Pearson"], alpha=1, c="black", s=20)
# plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
# plt.xlabel("iStar — Visium data")
# plt.ylabel("iStar — Xenium data")
# plt.title("Pearson Correlation per Gene")
# sns.despine()

# for gene in genes_highlight:
#     if gene not in vis_s["Gene"].values:
#         continue
#     x = vis_s.loc[vis_s["Gene"] == gene, "Pearson"].values[0]
#     y = xen_s.loc[xen_s["Gene"] == gene, "Pearson"].values[0]
#     plt.annotate(gene, (x, y), xytext=(5, 5), textcoords='offset points',
#                  fontsize=12, color="green")

# plt.savefig(OUT_DIR / "istar_fig2b_scatterplot.svg", dpi=1000, bbox_inches="tight")
# plt.close()

# # ── Fig 2b — rMSE_range scatter ──────────────────────────────────────────────
# print("Plotting rMSE_range scatter …")
# highlight_mask = vis_s["Gene"].isin(genes_highlight)

# plt.figure(figsize=(10, 5))
# plt.scatter(vis_s["rMSE_range"], xen_s["rMSE_range"], alpha=1, c="black", s=20,
#             label="Other genes")
# plt.scatter(vis_s.loc[highlight_mask, "rMSE_range"],
#             xen_s.loc[highlight_mask, "rMSE_range"],
#             alpha=1, c="green", label="Highlighted genes")
# plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
# plt.xlabel("iStar — Visium data")
# plt.ylabel("iStar — Xenium data")
# plt.title("Normalised rMSE (range) per Gene")
# sns.despine()
# plt.savefig(OUT_DIR / "istar_fig2b_rmse.svg", dpi=1000, bbox_inches="tight")
# plt.close()

# ── Spatial plots (fig2c equivalent) ─────────────────────────────────────────
# Build a patch-dict from adata_pred that is compatible with plotRasterSideBySide:
# each entry is a minimal AnnData with .X (1 × n_genes), .var_names, .uns['patch_coords'].

def build_pred_patch_dict(adata_pred, gt_patch_dict):
    """Convert iStar adata_pred (patches × genes) into a plotRaster-compatible dict."""
    from scipy.sparse import csr_matrix
    pred_patches = {}
    for patch_id in adata_pred.obs_names:
        if patch_id not in gt_patch_dict:
            continue
        row = adata_pred.X_array.loc[patch_id].values.reshape(1, -1)  # (1, n_genes)
        a = ad.AnnData(
            X=csr_matrix(row),
            obs=pd.DataFrame(index=[patch_id]),
            var=pd.DataFrame(index=adata_pred.var_names),
        )
        a.uns['patch_coords'] = gt_patch_dict[patch_id].uns['patch_coords']
        pred_patches[patch_id] = a
    return pred_patches


def _get_vals(patches, gene_name):
    """Return (values_array, gene_index) for a patch dict."""
    vals = []
    for p in patches.values():
        try:
            idx = list(p.var_names).index(gene_name)
            vals.append(float(p.X[:, idx].sum()))
        except ValueError:
            vals.append(np.nan)
    return np.array(vals)


def plotRasterSideBySide_istar(image, gt_patches, pred_patches, gene_name, save_path):
    """
    Side-by-side ground truth (left) vs iStar prediction (right).
    Both panels share a single colorbar using the 1st–99th percentile of the
    combined gt + pred values — same as figure_2.py's plotRasterSideBySide.
    """
    gt_vals   = _get_vals(gt_patches,   gene_name)
    pred_vals = _get_vals(pred_patches, gene_name)

    all_vals = np.concatenate([gt_vals, pred_vals])
    vlo = np.nanpercentile(all_vals, 1)
    vhi = np.nanpercentile(all_vals, 99)
    if vhi <= vlo:
        vhi = vlo + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, patches, title in zip(axes,
                                   [gt_patches, pred_patches],
                                   ["Ground Truth", "iStar Prediction"]):
        ax.imshow(image)
        for p in patches.values():
            x0, x1, y0, y1 = p.uns['patch_coords']
            try:
                idx = list(p.var_names).index(gene_name)
                val = float(p.X[:, idx].sum())
            except ValueError:
                val = vlo
            nv    = np.clip((val - vlo) / (vhi - vlo), 0, 1)
            color = plt.cm.viridis(nv)
            rect  = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        linewidth=0, facecolor=color, alpha=1)
            ax.add_patch(rect)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    norm = plt.Normalize(vlo, vhi)
    sm   = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.03, pad=0.04, shrink=0.75)
    cbar.set_label(f"{gene_name} Expression (log1p)", fontsize=12)

    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


print("\nBuilding spatial patch dicts …")
from PIL import Image as PILImage
PILImage.MAX_IMAGE_PIXELS = None   # large WSIs

# -- Visium spatial --
print("  Loading Visium adata and patches …")
adata_vis = sc.read_h5ad(ALIGNED_VIS / "visiumdata_visiumimage_data.h5ad")
sc.pp.log1p(adata_vis)
with open(ALIGNED_VIS / "visiumdata_visiumimage_patches.pkl", "rb") as f:
    vis_gt_patches = pickle.load(f)
# log1p ground truth (matching figure_2.py)
for k in vis_gt_patches:
    vis_gt_patches[k].X = sc.pp.log1p(np.round(vis_gt_patches[k].X.copy()))

adata_vis_pred = sc.read_h5ad(ISTAR_OUT / "visium/adata_pred.h5ad")
if not hasattr(adata_vis_pred, 'X_array') or adata_vis_pred.X_array is None:
    adata_vis_pred.X_array = pd.DataFrame(
        adata_vis_pred.X.toarray(),
        index=adata_vis_pred.obs_names,
        columns=adata_vis_pred.var_names,
    )
vis_pred_patches = build_pred_patch_dict(adata_vis_pred, vis_gt_patches)
vis_image = adata_vis.uns['spatial']

# -- Xenium spatial --
print("  Loading Xenium adata and patches …")
adata_xen = sc.read_h5ad(ALIGNED_XEN / "xeniumdata_xeniumimage_data.h5ad")
sc.pp.log1p(adata_xen)
with open(ALIGNED_XEN / "xeniumdata_xeniumimage_patches.pkl", "rb") as f:
    xen_gt_patches = pickle.load(f)
for k in xen_gt_patches:
    xen_gt_patches[k].X = sc.pp.log1p(np.round(xen_gt_patches[k].X.copy()))

adata_xen_pred = sc.read_h5ad(ISTAR_OUT / "xenium/adata_pred.h5ad")
if not hasattr(adata_xen_pred, 'X_array') or adata_xen_pred.X_array is None:
    adata_xen_pred.X_array = pd.DataFrame(
        adata_xen_pred.X.toarray(),
        index=adata_xen_pred.obs_names,
        columns=adata_xen_pred.var_names,
    )
xen_pred_patches = build_pred_patch_dict(adata_xen_pred, xen_gt_patches)
xen_image = adata_xen.uns['spatial']

# -- Plot for selected genes --
genes_spatial = ["HDC", "GZMK", "AHSP", "ANKRD30A"]
print("\nPlotting spatial maps …")
for gene in genes_spatial:
    if gene in adata_vis_pred.var_names:
        plotRasterSideBySide_istar(
            vis_image, vis_gt_patches, vis_pred_patches,
            gene_name=gene,
            save_path=OUT_DIR / f"istar_fig2c_{gene}_visium.svg",
        )
    else:
        print(f"  Skipping {gene} (not in Visium iStar output)")

    if gene in adata_xen_pred.var_names:
        plotRasterSideBySide_istar(
            xen_image, xen_gt_patches, xen_pred_patches,
            gene_name=gene,
            save_path=OUT_DIR / f"istar_fig2c_{gene}_xenium.svg",
        )
    else:
        print(f"  Skipping {gene} (not in Xenium iStar output)")

print("\nAll plots saved to:", OUT_DIR)



