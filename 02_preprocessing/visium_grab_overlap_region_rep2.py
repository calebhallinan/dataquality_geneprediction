
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl
import os
from scipy import stats
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import re
from torch.utils.data import Dataset
from mamba_ssm import Mamba
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import itertools
import shap
import pickle 


############################################################################################################


### Read in the Data ###



# file name
file_name = "breastcancer_xenium_sample1_rep2"
# resolution
resolution = 250
# read in the data
adata_xenium = sc.read_10x_h5('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep2/cell_feature_matrix.h5')

# Load the full-resolution spatial data
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_visium_high_res_STalign.csv.gz", index_col=0)
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/xenium_cell_centroids_visium_high_res.csv")
cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_fullresolution_STalign.csv.gz", index_col=0)
cell_centers

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image"
img = np.array(Image.open("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/" + file_name + "/" + img_name + ".tif"))
# img = np.load("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/visium_high_res_image.npy")
# plt.imshow(img)

# add .obs
adata_xenium.obs = cell_centers
# add .obsm
adata_xenium.obsm["spatial"] = adata_xenium.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata_xenium.uns['spatial'] = img
# need to add this for subsetting
adata_xenium.obs.index = adata_xenium.obs.index.astype(str)


# get rid of genes that aren't in visium
# gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
# gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]

# # subset the data # NOTE doing this for probe project bc we want everything
# adata_xenium = adata_xenium[:, gene_list]

# make an array of the gene expression data
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)

# need to subset bc there are negative values
adata_xenium = adata_xenium[adata_xenium.obs["y_centroid"] > 0]
adata_xenium = adata_xenium[adata_xenium.obs["x_centroid"] > 0]


# NO NORMALIZATION HERE


# plot the data
plt.figure(figsize=(18, 10))
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1], s=1, c="yellow")
plt.axis("off")


############################################################################################################


# should be the name of image data in adata
tissue_section = "CytAssist_FFPE_Human_Breast_Cancer"

# file path where outs data is located
file_path = "/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/"


# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
len(gene_list)

### Read in adata ###

# read data
adata_visium = sc.read_visium(file_path)
# make unique
adata_visium.var_names_make_unique()
# get mitochondrial gene expression info
adata_visium.var["mt"] = adata_visium.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata_visium, qc_vars=["mt"], inplace=True)


# make spatial position str to integer
# https://discourse.scverse.org/t/data-fomr-new-spatial-transcriptomics-from-10x/1107/6
# adata_visium.obsm['spatial'] = adata_visium.obsm['spatial'].astype(int)

# get new coordiantes
visium_aligned_coords = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/aligned_visium_points_to_xenium2_image.npy")
# add the new coordinates
adata_visium.obsm['spatial'] = visium_aligned_coords.astype(int)

# add the image
adata_visium.uns['spatial'] = img

# need to add this for subsetting
adata_visium.obs.index = adata_visium.obs.index.astype(str)

# remove any negative values
adata_visium = adata_visium[adata_visium.obsm['spatial'][:,0] > 0]
adata_visium = adata_visium[adata_visium.obsm['spatial'][:,1] > 0]

# NO NORMALIZATION HERE

# subet gene list
# adata_visium = adata_visium[:, gene_list]

# plot the data
plt.imshow(adata_visium.uns['spatial'])
plt.scatter(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1], s=1, c="red")


############################################################################################################


# resolution
resolution = 250
# adata_patches_xenium = rasterizeGeneExpression_topatches(adata_xenium_in_overlap.uns['spatial'], adata_xenium_in_overlap, patch_size=resolution, aggregation='sum', visium=False)
# len(adata_patches_xenium)
adata_patches_visium = rasterizeGeneExpression_topatches_basedoncenters(adata_visium.uns['spatial'], adata_visium, adata_visium.obsm['spatial'], patch_size=resolution, aggregation='sum', visium=True)
len(adata_patches_visium)

# now get the patches for xenium based on the centers of the visium patches
visium_coords = []
# make an array of the coordinates
for i in adata_patches_visium:
   visium_coords.append(list(adata_patches_visium[i].obsm["spatial"][0]))


### getting rid of the patches that are not in the overlap region ###

# Convert list-of-lists to list-of-tuples
visium_tuples = [tuple(coord) for coord in visium_coords]
# Convert obsm['spatial'] (n-by-2 array) to a DataFrame
df_spatial = pd.DataFrame(adata_visium.obsm['spatial'], columns=['x', 'y'])
# Turn each row into a tuple
spatial_tuples = df_spatial.apply(tuple, axis=1)  # e.g. (10.4, 5.2)
# Check membership using .isin()
mask = spatial_tuples.isin(visium_tuples)
# Subset your AnnData
adata_visium = adata_visium[mask, :]


# use new function to get patches based on centers
adata_patches_xenium = rasterizeGeneExpression_topatches_basedoncenters(adata_xenium.uns['spatial'], adata_xenium, visium_coords, patch_size=resolution, aggregation='sum', visium=False)
len(adata_patches_xenium)

# grab the coordinates of the patches
xenium_coords = []
# make an array of the coordinates
for i in adata_patches_xenium:
   xenium_coords.append(list(adata_patches_xenium[i].obsm["spatial"][0]))

# grab the coordinates of the patches
# visium_coords = []
# # make an array of the coordinates
# for i in adata_patches_visium:
#    visium_coords.append(list(adata_patches_visium[i].obsm["spatial"][0]))


# Create a mask based on whether each coordinate in visium_coords is in xenium_coords
def is_in_coords(coord, coords_array):
    return np.any(np.all(coords_array == coord, axis=1))

# Create a boolean mask for each coordinate in visium_coords
mask = np.array([is_in_coords(coord, np.array(xenium_coords)) for coord in np.array(visium_coords)])
len(xenium_coords)
len(visium_coords)


# Use the mask to subset the AnnData object
adata_visium_in_overlap = adata_visium[mask].copy()

# plot a circle at each x y coordinate and overlap
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(adata_xenium.obsm['spatial'][:, 0], adata_xenium.obsm['spatial'][:, 1], s=1)
plt.scatter(adata_visium_in_overlap.obsm['spatial'][:, 0], adata_visium_in_overlap.obsm['spatial'][:, 1], s=1, c="red")

# redo to get the patches for visium
adata_patches_visium_overlap = rasterizeGeneExpression_topatches_basedoncenters(adata_visium_in_overlap.uns['spatial'], adata_visium_in_overlap, adata_visium_in_overlap.obsm['spatial'], patch_size=resolution, aggregation='sum', visium=True)
len(adata_patches_visium_overlap)


# plot to confirm
plotRaster(adata_xenium.uns['spatial'], adata_patches_xenium, color_by='total_expression')
plotRaster(adata_xenium.uns['spatial'], adata_patches_xenium, color_by='gene_expression', gene_name="TOMM7")
plotRaster(adata_visium_in_overlap.uns['spatial'], adata_patches_visium_overlap, color_by='total_expression')
# plotRaster(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_patches_visium, color_by='total_expression')
plotRaster(adata_visium_in_overlap.uns['spatial'], adata_patches_visium_overlap, color_by='gene_expression', gene_name="HDC")


# sc.pl.spatial(adata_visium_in_overlap, img=adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], color=["LUM"], scale_factor = 1, size=adata_visium_in_overlap.uns['spatial'][tissue_section]['scalefactors']['tissue_hires_scalef'],frameon= False, title = "LUM")
# sc.pl.spatial(adata_visium, img=adata_visium.uns['spatial'][tissue_section]["images"]["hires"], color=["LUM"], scale_factor = 1, size=adata_visium.uns['spatial'][tissue_section]['scalefactors']['tissue_hires_scalef'],frameon= False, title = "LUM")

### need to make sure each patch has the same key in both dictionaries ###


# Extract coordinates from the new dictionaries
coords_visium = np.array([adata_patches_visium[i].obsm['spatial'][0] for i in adata_patches_visium])
coords_xenium = np.array([adata_patches_xenium[i].obsm['spatial'][0] for i in adata_patches_xenium])

# Create a mapping from coordinates to keys for both dictionaries
def create_coord_to_key_mapping(coords, adata_dict):
    coord_to_key = {}
    for key in adata_dict:
        coord = tuple(adata_dict[key].obsm['spatial'][0])
        coord_to_key[coord] = key
    return coord_to_key

# Map coordinates to keys for both dictionaries
coord_to_key_visium = create_coord_to_key_mapping(coords_visium, adata_patches_visium)
coord_to_key_xenium = create_coord_to_key_mapping(coords_xenium, adata_patches_xenium)

# Create new dictionaries with aligned keys
aligned_visium = {}
aligned_xenium = {}

for coord, key_visium in coord_to_key_visium.items():
    if coord in coord_to_key_xenium:
        key_xenium = coord_to_key_xenium[coord]
        aligned_visium[key_xenium] = adata_patches_visium[key_visium]
        aligned_xenium[key_xenium] = adata_patches_xenium[key_xenium]

# The aligned dictionaries now have the same keys based on spatial coordinates

# plt.imshow(aligned_visium["patch_0"].uns['spatial'])
# plt.imshow(aligned_xenium["patch_0"].uns['spatial'])
# len(aligned_xenium)
# len(aligned_visium)


# plt.imshow(aligned_xenium["patch_1503"].uns['spatial'])
# aligned_xenium["patch_1503"].obsm['spatial']


# combine the adata patches
combined_aligned_visium = combine_adata_patches(aligned_visium, adata_visium.uns['spatial'])
combined_aligned_xenium = combine_adata_patches(aligned_xenium, adata_xenium.uns['spatial'])


# plot a circle at each x y coordinate and overlap
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(combined_aligned_visium.obsm['spatial'][:, 0], combined_aligned_visium.obsm['spatial'][:, 1], s=1)
plt.scatter(combined_aligned_xenium.obsm['spatial'][:, 0], combined_aligned_xenium.obsm['spatial'][:, 1], s=1, c="red")
# plt.scatter(adata_visium_in_overlap.obsm['spatial'][:, 0], adata_visium_in_overlap.obsm['spatial'][:, 1], s=1, c="green")


plotRaster(combined_aligned_xenium.uns['spatial'], aligned_xenium, color_by='gene_expression', gene_name="KRT8")
plotRaster(combined_aligned_visium.uns['spatial'], aligned_visium, color_by='gene_expression', gene_name="KRT8")
plotRaster(combined_aligned_visium.uns['spatial'], aligned_visium, color_by='gene_expression', gene_name="TUBB2A")
# cant see them so this is good!


# plot a circle at each x y coordinate and overlap
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(combined_aligned_visium.obsm['spatial'][:, 0], combined_aligned_visium.obsm['spatial'][:, 1], s=1)
# plt.scatter(combined_aligned_xenium.obsm['spatial'][:8, 0], combined_aligned_xenium.obsm['spatial'][:8, 1], s=1, c="red")


####################################################################################################
####################################################################################################



# save all the data
combined_aligned_visium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/visium_data_full.h5ad")
combined_aligned_xenium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/xenium_data_full.h5ad")

# save the patches aligned_visium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/visium_patches_full.pkl', 'wb') as f:
    pickle.dump(aligned_visium, f)

# # save the patches aligned_xenium
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/xenium_patches_full.pkl', 'wb') as f:
    pickle.dump(aligned_xenium, f)













####################################################################################################

### CODE FOR PERMUTATION ###

# # function to split the adata patches
# def split_adata_patches(combined_adata, adata_patches_for_image):
#     # Dictionary to store the split AnnData objects
#     adata_patches = {}
    
#     # Get the unique keys (patch identifiers) from the obs index
#     unique_keys = combined_adata.obs.index.unique()
    
#     # Iterate over the unique keys to split the data
#     for key in unique_keys:
#         # Subset the combined AnnData for each key
#         adata_patch = combined_adata[combined_adata.obs.index == key].copy()
#         # Reset the obs index to default to avoid confusion
#         adata_patch.obs.index = adata_patch.obs.index.str.split('-').str[-1]

#         adata_patch.uns['spatial'] = adata_patches_for_image[key].uns['spatial']
#         # Store the individual AnnData object in the dictionary
#         adata_patches[key] = adata_patch
    
#     return adata_patches


# # Set the seed
# np.random.seed(42)

# arr_visium = combined_aligned_visium.X.toarray()
# arr_xenium = combined_aligned_xenium.X.toarray()

# # Permute values within each column using the same permutation
# for i in range(arr_visium.shape[1]):
#     perm = np.random.permutation(arr_visium[:, i])  # Generate permutation once
#     arr_visium[:, i] = perm
#     arr_xenium[:, i] = perm  # Apply the same permutation to arr_xenium

# # Update the AnnData objects
# combined_aligned_visium.X = arr_visium
# combined_aligned_xenium.X = arr_xenium


# # Split the combined AnnData objects into individual patches
# adata_patches_visium = split_adata_patches(combined_aligned_visium, aligned_visium)
# len(adata_patches_visium)
# adata_patches_xenium = split_adata_patches(combined_aligned_xenium, aligned_xenium)
# len(adata_patches_visium)




# def plotRaster1(image, adata_patches, resolution, color_by='gene_expression', gene_name=None):
#     """
#     Plots patches on the original image, colored by either gene expression or a column in adata_patches.obs.

#     Parameters:
#     - image: The original image array.
#     - adata_patches: Dictionary of AnnData objects representing the patches.
#     - resolution: Resolution value that determines the size of each patch.
#     - color_by: How to color the patches ('gene_expression' or 'total_expression').
#     - gene_name: The name of the gene to use if color_by is 'gene_expression'.
#     """
#     # Check inputs
#     if color_by == 'gene_expression' and gene_name is None:
#         raise ValueError("You must specify a gene_name when color_by='gene_expression'.")

#     # Collect all values for normalization
#     values = []
#     for adata_patch in adata_patches.values():
#         if color_by == 'gene_expression':
#             expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
#             values.append(expression)
#         elif color_by == 'total_expression':
#             total_expression = adata_patch.X.sum()
#             values.append(total_expression)
    
#     # Get min and max values
#     values = np.array(values)
#     min_value, max_value = values.min(), values.max()

#     # Plot the original image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image)

#     # Plot each patch with the appropriate color
#     for adata_patch in adata_patches.values():
#         # Calculate patch coordinates based on .obsm['spatial']
#         x_center, y_center = adata_patch.obsm['spatial'][0]  # Assuming single entry per patch
#         x_start = x_center - resolution / 2
#         x_end = x_center + resolution / 2
#         y_start = y_center - resolution / 2
#         y_end = y_center + resolution / 2
        
#         # Store the patch coordinates in .uns
#         adata_patch.uns['patch_coords'] = (x_start, x_end, y_start, y_end)
        
#         # Get the color value
#         if color_by == 'gene_expression':
#             expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
#             normalized_value = (expression - min_value) / (max_value - min_value)
#             color = plt.cm.viridis(normalized_value)
#         elif color_by == 'total_expression':
#             total_expression = adata_patch.X.sum()
#             normalized_value = (total_expression - min_value) / (max_value - min_value)
#             color = plt.cm.viridis(normalized_value)
        
#         # Draw a rectangle for the patch
#         rect = mpatches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
#                                   linewidth=1, edgecolor='none', facecolor=color, alpha=1)
#         ax.add_patch(rect)

#     # Create a color bar
#     norm = plt.Normalize(min_value, max_value)
#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
#     cbar.set_label(f'{gene_name} Expression' if color_by == 'gene_expression' else "Total Expression")

#     plt.axis('off')
#     plt.show()

# # plot to confirm
# plotRaster1(adata_xenium.uns['spatial'], adata_patches_xenium, resolution, color_by='total_expression')
# plotRaster1(adata_xenium.uns['spatial'], adata_patches_xenium, resolution, color_by='gene_expression', gene_name="TOMM7")
# plotRaster1(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_patches_visium, resolution, color_by='total_expression')
# plotRaster1(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_patches_visium, resolution, color_by='gene_expression', gene_name="TOMM7")


### for permuatioin ###

# # save the patches aligned_visium
# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_visium_dictionary_raw_permutation.pkl', 'wb') as f:
#     pickle.dump(adata_patches_visium, f)

# # save the patches aligned_xenium
# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw_permutation.pkl', 'wb') as f:
#     pickle.dump(adata_patches_xenium, f)



