
# import necessary packages
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable PIL's pixel limit


### read in data ###

###################################################################################################################

adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/HCC/xenium_adata_HCC.h5ad')


# read in image
image_xenium = Image.open("/home/caleb/Desktop/improvedgenepred/data/HCC/xenium_image_HCC.tif")
image_xenium_array = np.array(image_xenium)
image_xenium_array.shape

adata_xenium.uns['H&E resolution'][0]

adata_xenium.obsm['spatial']

# pixel per micron = 4.53


plt.imshow(image_xenium_array)
plt.scatter(adata_xenium.obsm['spatial'][:,0]/adata_xenium.uns['H&E resolution'][0], adata_xenium.obsm['spatial'][:,1]/adata_xenium.uns['H&E resolution'][1], s=0.5, alpha=.1, c='yellow')
# plt.scatter(adata_xenium.obsm['spatial'][:,0], adata_xenium.obsm['spatial'][:,1], s=0.5, alpha=.009, c='yellow')


###################################################################################################################


adata_cosmx = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/HCC/cosmx_adata_HCC.h5ad')


image_cosmx = Image.open("/home/caleb/Desktop/improvedgenepred/data/HCC/cosmx_image_HCC.tif")
image_cosmx_array = np.array(image_cosmx)
image_cosmx_array.shape

# pixel per micron = 4.53

plt.imshow(image_cosmx_array)
plt.scatter(adata_cosmx.obsm['spatial'][:,0]/adata_cosmx.uns['H&E resolution'][0], adata_cosmx.obsm['spatial'][:,1]/adata_cosmx.uns['H&E resolution'][0], s=0.5, alpha=.1, c='yellow')




###################################################################################################################


adata_visiumhd = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/HCC/visiumhd_adata_HCC.h5ad')

image_visiumhd = Image.open("/home/caleb/Desktop/improvedgenepred/data/HCC/visiumhd_image_HCC.tif")
image_visiumhd_array = np.array(image_visiumhd)
image_visiumhd_array.shape

# # pixel per micron = 3.81

plt.imshow(image_visiumhd_array)
plt.scatter(adata_visiumhd.obsm['spatial'][:,0]/adata_visiumhd.uns['H&E resolution'], adata_visiumhd.obsm['spatial'][:,1]/adata_visiumhd.uns['H&E resolution'], s=0.5, alpha=.1, c='yellow')




###################################################################################################################


adata_stereoseq = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/HCC/stereoseq_adata_HCC.h5ad')

image_stereoseq = Image.open("/home/caleb/Desktop/improvedgenepred/data/HCC/stereoseq_image_HCC.tif")
image_stereoseq_array = np.array(image_stereoseq)
image_stereoseq_array.shape

# pixel per micron = 0.2?


plt.imshow(image_stereoseq_array)
plt.scatter(adata_stereoseq.obsm['spatial'][:,0]/adata_stereoseq.uns['H&E resolution'], adata_stereoseq.obsm['spatial'][:,1]/adata_stereoseq.uns['H&E resolution'], s=0.5, alpha=.1, c='yellow')



###################################################################################################################



# ── Resolution for each dataset ────────────────────────────────────────────
resolutions = {
    "xenium":     adata_xenium.uns['H&E resolution'][0],      # µm/px
    "cosmx":      adata_cosmx.uns['H&E resolution'][0],       # µm/px
    "visiumhd":   float(adata_visiumhd.uns['H&E resolution']),# µm/px
    "stereoseq":  adata_stereoseq.uns['H&E resolution'],   # µm/px
}

patch_sizes_px = {
    "xenium":    250,
    "cosmx":     250,
    "visiumhd":  210,
    "stereoseq": 250,
}

print(f"{'Dataset':<12} {'µm/px':>8} {'patch_px':>10} {'patch_µm':>10}")
print("-" * 44)
for name, um_per_px in resolutions.items():
    px = patch_sizes_px[name]
    um = px * um_per_px
    print(f"{name:<12} {um_per_px:>8.4f} {px:>10} {um:>10.2f}")


target_um = 55.0

patch_sizes_px = {
    name: int(round(target_um / um_per_px))
    for name, um_per_px in resolutions.items()
}

patch_sizes_px

###################################################################################################################

# Find the intersection of the gene lists
xenium_genes = set(adata_xenium.var['gene_ids'].index)
cosmx_genes = set(adata_cosmx.var.index)
visiumhd_genes = set(adata_visiumhd.var['gene_ids'].index)
stereoseq_genes = set(adata_stereoseq.var.index)

common_genes = list(xenium_genes & cosmx_genes & visiumhd_genes & stereoseq_genes)
print(f"Number of common genes: {len(common_genes)}")
print(common_genes)

# save to csv
# pd.Series(common_genes).to_csv("shared_genes_visiumhd_xenium_cosmx_stereoseq_HCC.csv")



### explore gene expression ###

import numpy as np
import scipy.sparse as sp

def gene_variance(X):
    """
    Compute per-gene variance for dense or sparse matrices.
    Returns a 1D numpy array of length n_genes.
    """
    if sp.issparse(X):
        # E[X]
        mean = np.asarray(X.mean(axis=0)).ravel()
        # E[X^2]
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        var = X.var(axis=0)

    return var

var_xenium   = gene_variance(adata_xenium.X)
var_cosmx    = gene_variance(adata_cosmx.X)
var_visiumhd = gene_variance(adata_visiumhd.X)



# Find top 2000 highly variable genes from common genes
common_genes_set = set(common_genes)

# Get variance for common genes
var_xenium_common = np.array([var_xenium[i] if adata_xenium.var.index[i] in common_genes_set else -np.inf 
                               for i in range(len(var_xenium))])
var_cosmx_common = np.array([var_cosmx[i] if adata_cosmx.var.index[i] in common_genes_set else -np.inf 
                              for i in range(len(var_cosmx))])
var_visiumhd_common = np.array([var_visiumhd[i] if adata_visiumhd.var.index[i] in common_genes_set else -np.inf 
                                 for i in range(len(var_visiumhd))])

# Use mean variance across datasets
gene_var_mean = {}
for gene in common_genes:
    xenium_idx = adata_xenium.var.index.get_loc(gene) if gene in adata_xenium.var.index else None
    cosmx_idx = adata_cosmx.var.index.get_loc(gene) if gene in adata_cosmx.var.index else None
    visiumhd_idx = adata_visiumhd.var.index.get_loc(gene) if gene in adata_visiumhd.var.index else None
    
    variances = []
    if xenium_idx is not None:
        variances.append(var_xenium[xenium_idx])
    if cosmx_idx is not None:
        variances.append(var_cosmx[cosmx_idx])
    if visiumhd_idx is not None:
        variances.append(var_visiumhd[visiumhd_idx])
    
    gene_var_mean[gene] = np.mean(variances)

# Get top 2000 highly variable genes
top_2000_hvgs = sorted(gene_var_mean.items(), key=lambda x: x[1], reverse=True)[:2000]
top_2000_hvgs_names = [gene for gene, var in top_2000_hvgs]

print(f"Number of top HVGs: {len(top_2000_hvgs_names)}")
pd.Series(top_2000_hvgs_names).to_csv("shared_genes_visiumhd_xenium_cosmx_stereoseq_HCC.csv", index=False)


###################################################################################################################




