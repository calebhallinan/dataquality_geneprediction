
# import necessary packages
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import re
# from utils import plot_xenium_with_centers, extract_and_subset_patches, subset_by_patch
from squidpy.im import ImageContainer

# import packages
import scanpy as sc
# import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import anndata as ad
from squidpy.im import ImageContainer
# from PIL import ImageFile, Image
from utils import *
import os
import random
import re
import pickle




############################################################################################################

# # should be the name of image data in adata
# tissue_section = "CytAssist_FFPE_Human_Breast_Cancer"

# # file path where outs data is located
# file_path = "/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/"


# # read in svg results
# gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
# gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# # these were not in the data
# gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
# len(gene_list)

# ### Read in adata ###

# # read data
# adata_visium = sc.read_visium(file_path)
# # make unique
# adata_visium.var_names_make_unique()
# # get mitochondrial gene expression info
# adata_visium.var["mt"] = adata_visium.var_names.str.startswith("MT-")
# sc.pp.calculate_qc_metrics(adata_visium, qc_vars=["mt"], inplace=True)

# # make spatial position str to integer
# # https://discourse.scverse.org/t/data-fomr-new-spatial-transcriptomics-from-10x/1107/6
# adata_visium.obsm['spatial'] = adata_visium.obsm['spatial'].astype(int)

# # scale data - CRUCIAL for doing when using the hires image to crop
# adata_visium.obsm['spatial'] = np.floor(adata_visium.obsm["spatial"].astype(np.int64) * adata_visium.uns['spatial'][tissue_section]["scalefactors"]["tissue_hires_scalef"]).astype(int)
# adata_visium.obsm['spatial'].shape

# # flip the image
# adata_visium.uns['spatial'][tissue_section]["images"]["hires"] = np.flip(adata_visium.uns['spatial'][tissue_section]["images"]["hires"], axis=0)

# # rotate the image
# adata_visium.uns['spatial'][tissue_section]["images"]["hires"] = np.rot90(adata_visium.uns['spatial'][tissue_section]["images"]["hires"], k=3)


# # function to transform coordinates
# def transform_coordinates(x, y, image_width, image_height, rotation_k=0, flip_axis=None):
#     """
#     Transforms (x, y) coordinates to match image transformations using np.flip first, then np.rot90.
    
#     Parameters:
#     - x: Array of original x-coordinates.
#     - y: Array of original y-coordinates.
#     - image_width: Width of the image.
#     - image_height: Height of the image.
#     - rotation_k: Number of 90-degree rotations counterclockwise (0, 1, 2, or 3).
#     - flip_axis: Axis to flip (None, 0 for vertical, 1 for horizontal).
    
#     Returns:
#     - x_new: Transformed x-coordinates (array of integers).
#     - y_new: Transformed y-coordinates (array of integers).
#     """
#     # Ensure x and y are numpy arrays
#     x = np.asarray(x)
#     y = np.asarray(y)

#     # Step 1: Apply flipping using np.flip
#     if flip_axis == 0:  # Vertical flip
#         y = image_height - 1 - y
#     elif flip_axis == 1:  # Horizontal flip
#         x = image_width - 1 - x

#     # Step 2: Apply rotation using np.rot90
#     if rotation_k % 4 == 1:  # 90 degrees counterclockwise
#         x_new = image_height - 1 - y
#         y_new = x
#     elif rotation_k % 4 == 2:  # 180 degrees
#         x_new = image_width - 1 - x
#         y_new = image_height - 1 - y
#     elif rotation_k % 4 == 3:  # 270 degrees counterclockwise (90 degrees clockwise)
#         x_new = y
#         y_new = image_width - 1 - x
#     else:  # rotation_k % 4 == 0, no rotation
#         x_new, y_new = x, y

#     # Ensure the final coordinates are integers
#     x_new = np.round(x_new).astype(int)
#     y_new = np.round(y_new).astype(int)

#     return x_new, y_new


# # Apply the transformation to x and y centroids
# adata_visium.obsm['spatial'] = np.array(transform_coordinates(
#     adata_visium.obsm['spatial'][:, 0],
#     adata_visium.obsm['spatial'][:, 1],
#     adata_visium.uns['spatial'][tissue_section]["images"]["hires"].shape[1],  # Width comes first for x-coordinates
#     adata_visium.uns['spatial'][tissue_section]["images"]["hires"].shape[0],  # Height comes second for y-coordinates
#     rotation_k=1,
#     flip_axis=0  # Vertical flip
# )).T

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_data.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_patches.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)


# scale the data and log transform
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# scale the data and log transform
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))

# log transform the data
# sc.pp.log1p(adata_xenium)
# sc.pp.log1p(adata_visium)

# plot to confirm
g = "ABCC11"
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)

############################################################################################################


### READ IN GENE EXPRESSION DATA ###


# make .X a dataframe to save
x = pd.DataFrame(adata_visium.X.toarray(), columns=adata_visium.var_names, index=adata_visium.obs.index)
# save gt
x.T.to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_gt.csv", index_label="gene")

# remove index patch_1587 because there is no data for it
patch_1587 = x.loc["patch_1587"]
x = x.drop("patch_1587", axis=0)
# save the data
x.T.to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_gt_4KNN.csv", index_label="gene")


############################################################################################################

### KNN Smoothing ###

# https://github.com/yanailab/knn-smoothing/tree/master
# https://www.biorxiv.org/content/10.1101/217737v3.full

# run knn smoothing
!python3 knn_smooth.py -k 50 -d 20 --sep , -f /home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_gt_4KNN.csv -o /home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_KNN.csv

# read in smoothed data
smoothed_x = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_KNN.csv", index_col=0)
# add back patch_1587 in index 1587 of the dataframe
smoothed_x = pd.concat([smoothed_x.iloc[:,:1274], patch_1587, smoothed_x.iloc[:,1274:]], axis=1)

# change x to smoothed x
adata_visium_KNN = adata_visium.copy()
adata_visium_KNN.X = scipy.sparse.csr_matrix(smoothed_x.T)

# make new patches
aligned_visium_dictionary_KNN = rasterizeGeneExpression_topatches_basedoncenters(adata_visium_KNN.uns['spatial'], adata_visium_KNN, adata_visium_KNN.obsm['spatial'], patch_size=250, aggregation='sum', visium=False)

# replace patch key with original patch key from aligned_xenium_dictionary
aligned_visium_dictionary_KNN_new = {}
for i in range(len(aligned_visium_dictionary_KNN)):
    old_key = list(aligned_visium_dictionary_KNN.keys())[i]
    new_key = list(aligned_xenium_dictionary.keys())[i]
    aligned_visium_dictionary_KNN_new[new_key] = aligned_visium_dictionary_KNN[old_key]

# rename
aligned_visium_dictionary_KNN = aligned_visium_dictionary_KNN_new

# log patches
# for i in aligned_visium_dictionary_KNN:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_visium_dictionary_KNN[i].X = sc.pp.log1p(aligned_visium_dictionary_KNN[i].X)
#     # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# plot
plotRaster(adata_visium_KNN.uns['spatial'], aligned_visium_dictionary_KNN, color_by='gene_expression', gene_name=g)

# save
smoothed_x.to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_KNN.csv", index_label="gene")

adata_visium_KNN.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data_KNN.h5ad")

# save the patches aligned_visium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_KNN.pkl', 'wb') as f:
    pickle.dump(aligned_visium_dictionary_KNN, f)

############################################################################################################

### MAGIC Imputation ###


import magic
import pandas as pd
import matplotlib.pyplot as plt

# https://magic.readthedocs.io/en/stable/
# https://nbviewer.org/github/KrishnaswamyLab/magic/blob/master/python/tutorial_notebooks/emt_tutorial.ipynb

# make .X a dataframe to save
x = pd.DataFrame(adata_visium.X.toarray(), columns=adata_visium.var_names, index=adata_visium.obs.index)

# for magic, log transform the data
x = np.log1p(x)

# create magic operator
magic_operator = magic.MAGIC(knn=50, t = "auto", random_state=0)
# transform data
X_magic = magic_operator.fit_transform(x)

# change x to smoothed x
adata_visium_magic = adata_visium.copy()
adata_visium_magic.X = scipy.sparse.csr_matrix(X_magic)

# make new patches
adata_patches_visium_magic = rasterizeGeneExpression_topatches_basedoncenters(adata_visium_magic.uns['spatial'], adata_visium_magic, adata_visium_magic.obsm['spatial'], patch_size=250, aggregation='sum', visium=False)
len(adata_patches_visium_magic)

# replace patch key with original patch key from aligned_xenium_dictionary
adata_patches_visium_magic_new = {}
for i in range(len(adata_patches_visium_magic)):
    old_key = list(adata_patches_visium_magic.keys())[i]
    new_key = list(aligned_xenium_dictionary.keys())[i]
    adata_patches_visium_magic_new[new_key] = adata_patches_visium_magic[old_key]


# rename
adata_patches_visium_magic = adata_patches_visium_magic_new

# plot results
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium_magic.uns['spatial'], adata_patches_visium_magic, color_by='gene_expression', gene_name=g)

# save
X_magic.to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_MAGIC.csv", index_label="gene")

adata_visium_magic.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data_MAGIC.h5ad")

# save the patches aligned_visium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_MAGIC.pkl', 'wb') as f:
    pickle.dump(adata_patches_visium_magic, f)


############################################################################################################

### SCVI Imputation ###

import tempfile

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scipy.stats import spearmanr
from scvi.data import cortex, smfish
from scvi.external import GIMVI

# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/gimvi_tutorial.html
# NOTE: use xenium-publication-env

# set seed
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

# figure settings
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()
%config InlineBackend.print_figure_kwargs={"facecolor": "w"}
%config InlineBackend.figure_format="retina"

# grab the data
train_size = 0.8
# spatial_data = smfish(save_path=save_dir.name) # adata_visium
# seq_data = cortex(save_path=save_dir.name)
spatial_data = adata_visium.copy()
chrom = sc.read_10x_h5("/home/caleb/Desktop/improvedgenepred/data/breastcancer_chromium_frp/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5")
seq_data = chrom.copy()

# preprocessing needed
# make variable names unique
seq_data.var_names_make_unique()
# get rid of these genes in spatial data
genes_to_remove = ['TKT', 'SLC39A4', 'GABARAPL2']
spatial_data = spatial_data[:, [gene for gene in spatial_data.var_names if gene not in genes_to_remove]]


# only use genes in both datasets
seq_data = seq_data[:, list(spatial_data.var_names)].copy()

seq_gene_names = seq_data.var_names
n_genes = seq_data.n_vars
n_train_genes = int(n_genes * train_size)

# randomly select training_genes
rand_train_gene_idx = np.random.choice(range(n_genes), n_train_genes, replace=False)
rand_test_gene_idx = sorted(set(range(n_genes)) - set(rand_train_gene_idx))
rand_train_genes = seq_gene_names[rand_train_gene_idx]
rand_test_genes = seq_gene_names[rand_test_gene_idx]

# spatial_data_partial has a subset of the genes to train on
spatial_data_partial = spatial_data[:, rand_train_genes].copy()

# remove cells with no counts
# sc.pp.filter_cells(spatial_data_partial, min_counts=1)
# sc.pp.filter_cells(seq_data, min_counts=1)

# setup_anndata for spatial and sequencing data
GIMVI.setup_anndata(spatial_data_partial)
GIMVI.setup_anndata(seq_data)

# spatial_data should use the same cells as our training data
# cells may have been removed by scanpy.pp.filter_cells()
spatial_data = spatial_data[spatial_data_partial.obs_names]

model = GIMVI(seq_data, spatial_data_partial)
model.train(max_epochs=300)



# get the latent representations for the sequencing and spatial data
latent_seq, latent_spatial = model.get_latent_representation()

# concatenate to one latent representation
latent_representation = np.concatenate([latent_seq, latent_spatial])
latent_adata = anndata.AnnData(latent_representation)

# labels which cells were from the sequencing dataset and which were from the spatial dataset
latent_labels = (["seq"] * latent_seq.shape[0]) + (["spatial"] * latent_spatial.shape[0])
latent_adata.obs["labels"] = latent_labels

# compute umap
sc.pp.neighbors(latent_adata, use_rep="X")
sc.tl.umap(latent_adata)

# save umap representations to original seq and spatial_datasets
seq_data.obsm["X_umap"] = latent_adata.obsm["X_umap"][: seq_data.shape[0]]
spatial_data.obsm["X_umap"] = latent_adata.obsm["X_umap"][seq_data.shape[0] :]


# umap of the combined latent space
sc.pl.umap(latent_adata, color="labels", show=True)


# utility function for scoring the imputation


def imputation_score(model, data_spatial, gene_ids_test, normalized=True):

    _, fish_imputation = model.get_imputed_values(normalized=normalized)
    original, imputed = (
        data_spatial.X[:, gene_ids_test],
        fish_imputation[:, gene_ids_test],
    )

    if normalized:
        original = original / data_spatial.X.sum(axis=1).reshape(-1, 1)

    spearman_gene = []
    for g in range(imputed.shape[1]):
        if np.all(imputed[:, g] == 0):
            correlation = 0
        else:
            correlation = spearmanr(original.toarray()[:, g], imputed[:, g])[0]
        spearman_gene.append(correlation)
    return np.nanmedian(np.array(spearman_gene))


imputation_score(model, spatial_data, rand_test_gene_idx, True)



# impute the data
_, fish_imputation = model.get_imputed_values(normalized=False)
original, imputed = (
    spatial_data.X[:, ],
    fish_imputation[:, ],
)

# make new adata object with imputed data
spatial_data_imputed = spatial_data.copy()
spatial_data_imputed.X = fish_imputation


# plot to confirm
g = "TOMM7"
sc.pl.spatial(spatial_data, color=g, spot_size=10, cmap="Reds")
sc.pl.spatial(spatial_data_imputed, color=g, spot_size=10, cmap="Reds")


plt.imshow(spatial_data.uns['spatial'])

# make new patches
adata_patches_visium_SCVI = rasterizeGeneExpression_topatches_basedoncenters(spatial_data_imputed.uns['spatial'], spatial_data_imputed, spatial_data_imputed.obsm['spatial'], patch_size=250, aggregation='sum', visium=False)
len(adata_patches_visium_SCVI)

# replace patch key with original patch key from aligned_xenium_dictionary
adata_patches_visium_SCVI_new = {}
for i in range(len(adata_patches_visium_SCVI)):
    old_key = list(adata_patches_visium_SCVI.keys())[i]
    new_key = list(aligned_xenium_dictionary.keys())[i]
    adata_patches_visium_SCVI_new[new_key] = adata_patches_visium_SCVI[old_key]


# rename
adata_patches_visium_SCVI = adata_patches_visium_SCVI_new
del adata_patches_visium_SCVI_new

# plot results
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium.uns['spatial'], adata_patches_visium_SCVI, color_by='gene_expression', gene_name=g)


# save imputed data
pd.DataFrame(spatial_data_imputed.X, columns=spatial_data_imputed.var_names, index=spatial_data_imputed.obs.index).to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_SCVI.csv", index_label="gene")


spatial_data_imputed.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data_SCVI.h5ad")

# save the patches aligned_visium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_SCVI.pkl', 'wb') as f:
    pickle.dump(adata_patches_visium_SCVI, f)


############################################################################################################


### impute xenium data after making it more sparse ###

# NOTE: did this for sparse less than 3 and poisson lambda 10, just change the file names


# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_xenium_raw_poissonlambda10.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_visium_raw.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_visium_dictionary_raw.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw_poissonlambda10.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)


# # scale the data
# scaling_factor = 1
# for i in aligned_visium_dictionary:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
#     # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# # adata_visium.X_array = adata_visium.X_array * scaling_factor

# # scale the data
# scaling_factor = 1
# for i in aligned_xenium_dictionary:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
#     # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


from utils import plotRaster


# plot the data
g = "TOMM7"
# plot to confirm
plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)




### SCVI Imputation ###

import tempfile

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scipy.stats import spearmanr
from scvi.data import cortex, smfish
from scvi.external import GIMVI

# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/gimvi_tutorial.html
# NOTE: use xenium-publication-env

# set seed
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

# figure settings
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = tempfile.TemporaryDirectory()
%config InlineBackend.print_figure_kwargs={"facecolor": "w"}
%config InlineBackend.figure_format="retina"

# grab the data
train_size = 0.8
# spatial_data = smfish(save_path=save_dir.name) # adata_visium
# seq_data = cortex(save_path=save_dir.name)
spatial_data = adata_xenium.copy()
chrom = sc.read_10x_h5("/home/caleb/Desktop/improvedgenepred/data/breastcancer_chromium_frp/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5")
seq_data = chrom.copy()

# preprocessing needed
# make variable names unique
seq_data.var_names_make_unique()
# get rid of these genes in spatial data
genes_to_remove = ['TKT', 'SLC39A4', 'GABARAPL2']
spatial_data = spatial_data[:, [gene for gene in spatial_data.var_names if gene not in genes_to_remove]]


# only use genes in both datasets
seq_data = seq_data[:, list(spatial_data.var_names)].copy()

seq_gene_names = seq_data.var_names
n_genes = seq_data.n_vars
n_train_genes = int(n_genes * train_size)

# randomly select training_genes
rand_train_gene_idx = np.random.choice(range(n_genes), n_train_genes, replace=False)
rand_test_gene_idx = sorted(set(range(n_genes)) - set(rand_train_gene_idx))
rand_train_genes = seq_gene_names[rand_train_gene_idx]
rand_test_genes = seq_gene_names[rand_test_gene_idx]

# spatial_data_partial has a subset of the genes to train on
spatial_data_partial = spatial_data[:, rand_train_genes].copy()

# remove cells with no counts
sc.pp.filter_cells(spatial_data_partial, min_counts=1)
sc.pp.filter_cells(seq_data, min_counts=1)

# setup_anndata for spatial and sequencing data
GIMVI.setup_anndata(spatial_data_partial)
GIMVI.setup_anndata(seq_data)

# spatial_data should use the same cells as our training data
# cells may have been removed by scanpy.pp.filter_cells()
spatial_data = spatial_data[spatial_data_partial.obs_names]

model = GIMVI(seq_data, spatial_data_partial)
model.train(max_epochs=200)



# get the latent representations for the sequencing and spatial data
latent_seq, latent_spatial = model.get_latent_representation()

# concatenate to one latent representation
latent_representation = np.concatenate([latent_seq, latent_spatial])
latent_adata = anndata.AnnData(latent_representation)

# labels which cells were from the sequencing dataset and which were from the spatial dataset
latent_labels = (["seq"] * latent_seq.shape[0]) + (["spatial"] * latent_spatial.shape[0])
latent_adata.obs["labels"] = latent_labels

# compute umap
sc.pp.neighbors(latent_adata, use_rep="X")
sc.tl.umap(latent_adata)

# save umap representations to original seq and spatial_datasets
seq_data.obsm["X_umap"] = latent_adata.obsm["X_umap"][: seq_data.shape[0]]
spatial_data.obsm["X_umap"] = latent_adata.obsm["X_umap"][seq_data.shape[0] :]


# umap of the combined latent space
sc.pl.umap(latent_adata, color="labels", show=True)

# check correlation between imputed and original xenium
imputation_score(model, spatial_data, rand_test_gene_idx, True)



# impute the data
_, fish_imputation = model.get_imputed_values(normalized=False)
original, imputed = (
    spatial_data.X[:, ],
    fish_imputation[:, ],
)

# make new adata object with imputed data
spatial_data_imputed = spatial_data.copy()
spatial_data_imputed.X = fish_imputation

# log transform
# sc.pp.log1p(spatial_data_imputed)
# sc.pp.log1p(spatial_data)


# function to split the data
def split_adata_patches(combined_adata, adata_patches_for_image):
    # Dictionary to store the split AnnData objects
    adata_patches = {}
    
    # Get the unique keys (patch identifiers) from the obs index
    unique_keys = combined_adata.obs.index.unique()
    
    # Iterate over the unique keys to split the data
    for key in unique_keys:
        # Subset the combined AnnData for each key
        adata_patch = combined_adata[combined_adata.obs.index == key].copy()
        # Reset the obs index to default to avoid confusion
        adata_patch.obs.index = adata_patch.obs.index.str.split('-').str[-1]

        adata_patch.uns['spatial'] = adata_patches_for_image[key].uns['spatial']
        adata_patch.uns['patch_coords'] = adata_patches_for_image[key].uns['patch_coords']
        # Store the individual AnnData object in the dictionary
        adata_patches[key] = adata_patch
    
    return adata_patches


# split the data
adata_patches_xenium = split_adata_patches(spatial_data, aligned_xenium_dictionary)
adata_patches_xenium_imputed = split_adata_patches(spatial_data_imputed, aligned_xenium_dictionary)



# plot
g = "TOMM7"
plotRaster(spatial_data.uns['spatial'], adata_patches_xenium, color_by='gene_expression', gene_name=g)
plotRaster(spatial_data_imputed.uns['spatial'], adata_patches_xenium_imputed, color_by='gene_expression', gene_name=g)

# save imputed data
# pd.DataFrame(spatial_data_imputed.X, columns=spatial_data_imputed.var_names, index=spatial_data_imputed.obs.index).to_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/xenium_gene_expression_imputed_SCVI_after_sparselessthan3.csv", index_label="gene")

# check correlation between imputed and original xenium


# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_xenium_raw_sparse_lessthan3.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)

# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))

# log transform the data
sc.pp.log1p(adata_xenium)

# remove some cells
adata_xenium = adata_xenium[spatial_data_partial.obs_names]


g = "ABCC11"
# plot to confirm
plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(spatial_data.uns['spatial'], adata_patches_xenium, color_by='gene_expression', gene_name=g)
plotRaster(spatial_data_imputed.uns['spatial'], adata_patches_xenium_imputed, color_by='gene_expression', gene_name=g)



# get correlation
correlation = []
for i in range(len(adata_xenium.X.toarray().T)):
    correlation.append(spearmanr(adata_xenium.X.toarray()[:,i], spatial_data_imputed.X[:,i])[0])

# print correlation
np.nanmedian(correlation)


# save all the data
spatial_data_imputed.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_xenium_raw_SCVI_after_poissonlambda10.h5ad")


# save the patches aligned_xenium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw_SCVI_after_poissonlambda10.pkl', 'wb') as f:
    pickle.dump(adata_patches_xenium_imputed, f)
