
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
import cv2

############################################################################################################


### Read in the Data ###


# file name
file_name = "breastcancer_xenium_sample1_rep1"
# resolution
# resolution = 12
# read in the data
adata_xenium = sc.read_10x_h5('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/cell_feature_matrix.h5')

# Load the full-resolution spatial data
cell_centers = pd.read_csv(f"/home/caleb/Desktop/improvedgenepred/data/{file_name}/{file_name}_fullresolution_STalign.csv.gz", index_col=0)
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/xenium_cell_centroids_visium_high_res.csv")
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/scaled_spots_for_xenium_image.csv", index_col=0)
# cell_centers.columns = ["x_centroid", "y_centroid"]

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
img = np.array(Image.open("/home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))
# img = np.load("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/visium_high_res_image.npy")
# plt.imshow(img)



# change image quality with gaussian blur
gb = 75
img = cv2.GaussianBlur(img,(gb, gb), sigmaX=0)





# add .obs
adata_xenium.obs = cell_centers
# add .obsm
adata_xenium.obsm["spatial"] = adata_xenium.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata_xenium.uns['spatial'] = img
# need to add this for subsetting
adata_xenium.obs.index = adata_xenium.obs.index.astype(str)

# adata_xenium.X = np.arcsinh(adata_xenium.X).toarray()

# scale genes with cpm
# sc.pp.normalize_total(adata_xenium, target_sum=1e6)

# log transform the data
# sc.pp.log1p(adata_xenium)

# sc.pp.normalize_total(adata_xenium, target_sum=1e6)

# get rid of genes that aren't in visium
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
# subset the data
adata_xenium = adata_xenium[:, gene_list]

# make an array of the gene expression data
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)

# plt.imshow(adata_xenium.uns['spatial'])




############################################################################################################


# should be the name of image data in adata
tissue_section = "CytAssist_FFPE_Human_Breast_Cancer"

# file path where outs data is located
file_path = "/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/"


# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
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
# READ IN ALIGNED DATA
aligned_visium_points = np.round(np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/aligned_visium_points_to_xenium_image.npy")).astype(int)
adata_visium.obsm['spatial'] = aligned_visium_points

# normalize data
# sc.pp.normalize_total(adata, inplace=True) # NOTE: same scale, proportion
# scp.normalize.library_size_normalize(adata)
# sc.pp.normalize_total(adata, target_sum=1e6)
# sc.pp.log1p(adata_visium)
# sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1000)

# scale data - CRUCIAL for doing when using the hires image to crop
# adata_visium.obsm['spatial'] = np.floor(adata_visium.obsm["spatial"].astype(np.int64) * adata_visium.uns['spatial'][tissue_section]["scalefactors"]["tissue_hires_scalef"]).astype(int)
# adata_visium.obsm['spatial'].shape

# get new cell centers for high rez image
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/scaled_spots_for_xenium_image.csv", index_col=0)
# cell_centers.columns = ["x_centroid", "y_centroid"]
# adata_visium.obsm['spatial'] = cell_centers.to_numpy().astype(int)

# subet gene list
adata_visium = adata_visium[:, gene_list]

# read in xenuum image
Image.MAX_IMAGE_PIXELS = None
file_name = "breastcancer_xenium_sample1_rep1"
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
# img_array = np.array(Image.open("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/" + file_name + "/" + img_name + ".tif"))
# flip and rotate the image
# img_array = np.fliplr(img_array)
# img_array = np.rot90(img_array, k=1)

# set the hires image
adata_visium.uns['spatial'][tissue_section]["images"]["hires"] = img


# rotate the cell centers to align with the hires image
# adata_visium.obsm['spatial'] = np.flip(adata_visium.obsm['spatial'], axis=1)

sc.pl.spatial(adata_visium, img=img, color=["LUM"], scale_factor = 1, size=1, frameon= False, title = "LUM", alpha=.8)


############################################################################################################







#### TEsting how to get the correct size patches


# plt.imshow(adata_visium.uns['spatial'][tissue_section]["images"]["hires"])
# # plot a circle at each x y coordinate
# plt.scatter(adata_visium.obsm['spatial'][:, 0][0:5], adata_visium.obsm['spatial'][:, 1][0:5], s=5, c="red")

# sc.pl.spatial(adata_visium, img=img_array, color=["LUM"], scale_factor = 1, size=1.75, frameon= False, title = "LUM", alpha=.8)

# adata_visium.obsm['spatial'] = adata_visium.obsm['spatial'][np.lexsort((adata_visium.obsm['spatial'][:, 0], adata_visium.obsm['spatial'][:, 1]))]

# (22138-21665)



# from scipy.spatial import distance_matrix

# # Sample coordinates for demonstration (you should replace these with your actual coordinates)
# x = adata_visium.obsm['spatial'][:, 0]
# y = adata_visium.obsm['spatial'][:, 1]

# # Stack coordinates together
# coords = np.vstack((x, y)).T

# # Calculate the pairwise distance matrix
# distances = distance_matrix(coords, coords)

# # Set the diagonal to infinity to ignore self-distances
# np.fill_diagonal(distances, np.inf)

# # Find the minimum distance (this is the distance between the closest pair of circles)
# min_distance = np.min(distances)

# # Define the patch size to be slightly less than the minimum distance
# patch_size = min_distance * 0.99

# # Print out the results
# print(f"Minimum distance between centers: {min_distance}")
# print(f"Suggested patch size (no overlap): {patch_size}")

# # Visualization
# fig, ax = plt.subplots()
# ax.scatter(x, y, s=100, color='red', label='Circle Centers')

# # Draw the patches
# for coord in coords:
#     rect = plt.Rectangle((coord[0] - patch_size/2, coord[1] - patch_size/2), 
#                          patch_size, patch_size, linewidth=1, edgecolor='blue', facecolor='none')
#     ax.add_patch(rect)

# plt.gca().set_aspect('equal', adjustable='box')
# plt.legend()
# plt.show()








# first rasterize the gene expression data 

# resolution
resolution = 250
# adata_patches_xenium = rasterizeGeneExpression_topatches(adata_xenium_in_overlap.uns['spatial'], adata_xenium_in_overlap, patch_size=resolution, aggregation='sum', visium=False)
# len(adata_patches_xenium)
adata_patches_visium = rasterizeGeneExpression_topatches_basedoncenters(adata_visium.uns['spatial'][tissue_section]["images"]["hires"], adata_visium, adata_visium.obsm['spatial'], patch_size=resolution, aggregation='sum', visium=True)
len(adata_patches_visium)

# now get the patches for xenium based on the centers of the visium patches
visium_coords = []
# make an array of the coordinates
for i in adata_patches_visium:
   visium_coords.append(list(adata_patches_visium[i].obsm["spatial"][0]))

# use new function to get patches based on centers
adata_patches_xenium = rasterizeGeneExpression_topatches_basedoncenters(adata_xenium.uns['spatial'], adata_xenium, visium_coords, patch_size=resolution, aggregation='sum', visium=False)
len(adata_patches_xenium)

# grab the coordinates of the patches
xenium_coords = []
# make an array of the coordinates
for i in adata_patches_xenium:
   xenium_coords.append(list(adata_patches_xenium[i].obsm["spatial"][0]))

# grab the coordinates of the patches
visium_coords = []
# make an array of the coordinates
for i in adata_patches_visium:
   visium_coords.append(list(adata_patches_visium[i].obsm["spatial"][0]))


# Create a mask based on whether each coordinate in visium_coords is in xenium_coords
def is_in_coords(coord, coords_array):
    return np.any(np.all(coords_array == coord, axis=1))

# Create a boolean mask for each coordinate in visium_coords
mask = np.array([is_in_coords(coord, np.array(xenium_coords)) for coord in np.array(visium_coords)])


# Use the mask to subset the AnnData object
adata_visium_in_overlap = adata_visium[mask].copy()

# plot a circle at each x y coordinate and overlap
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(adata_xenium.obsm['spatial'][:, 0], adata_xenium.obsm['spatial'][:, 1], s=1)
plt.scatter(adata_visium_in_overlap.obsm['spatial'][:, 0], adata_visium_in_overlap.obsm['spatial'][:, 1], s=1, c="red")

# redo to get the patches for visium
adata_patches_visium_overlap = rasterizeGeneExpression_topatches_basedoncenters(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_visium_in_overlap, adata_visium_in_overlap.obsm['spatial'], patch_size=resolution, aggregation='sum', visium=True)
len(adata_patches_visium_overlap)


# plot to confirm
plotRaster(adata_xenium.uns['spatial'], adata_patches_xenium, color_by='total_expression')
plotRaster(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_patches_visium_overlap, color_by='total_expression')
plotRaster(adata_visium_in_overlap.uns['spatial'][tissue_section]["images"]["hires"], adata_patches_visium, color_by='total_expression')



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

plt.imshow(aligned_visium["patch_1000"].uns['spatial'])
plt.imshow(aligned_xenium["patch_1000"].uns['spatial'])
len(aligned_xenium)
len(aligned_visium)

# combine the adata patches
combined_aligned_visium = combine_adata_patches(aligned_visium, adata_visium.uns['spatial'][tissue_section]["images"]["hires"])
combined_aligned_xenium = combine_adata_patches(aligned_xenium, adata_xenium.uns['spatial'])


# plot a circle at each x y coordinate and overlap
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(combined_aligned_visium.obsm['spatial'][:, 0], combined_aligned_visium.obsm['spatial'][:, 1], s=1)
plt.scatter(combined_aligned_xenium.obsm['spatial'][:, 0], combined_aligned_xenium.obsm['spatial'][:, 1], s=1, c="red")
# plt.scatter(adata_visium_in_overlap.obsm['spatial'][:, 0], adata_visium_in_overlap.obsm['spatial'][:, 1], s=1, c="green")

# cant see them so this is good!

plt.imshow(aligned_visium['patch_1'].uns['spatial'])

# save all the data
combined_aligned_visium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_visium_raw_gb" + str(gb) + ".h5ad")
combined_aligned_xenium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_xenium_raw_gb" + str(gb) + ".h5ad")


# save the patches aligned_visium
with open("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_visium_dictionary_raw_gb" + str(gb) + ".pkl", "wb") as f:
    pickle.dump(aligned_visium, f)

# save the patches aligned_xenium
with open("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_xenium_dictionary_raw_gb" + str(gb) + ".pkl", "wb") as f:
    pickle.dump(aligned_xenium, f)



