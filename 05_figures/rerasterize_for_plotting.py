
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl
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
from adjustText import adjust_text  # Import the adjustText library


############################################################################################################

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
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)


# choose method
method = "visium"
# method = "xenium"

# img
image_version = "visium"
# image_version = "xenium"

# resolution
resolution = 250

# edits
edit = "none"

# prepare the datasets
if method == "visium":
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary)
else:
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary)


# plot the data
g = "TOMM7"
# plot to confirm
plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)



############################################################################################################



# rerasterize the data

# Function to plot patches on the original image
def plotRaster(image, adata_patches, color_by='gene_expression', gene_name=None, is_pred=False, data_type='visium'):
    """
    Plots patches on the original image, colored by either gene expression or a column in adata_patches.obs.

    Parameters:
    - image: The original image array.
    - adata_patches: Dictionary of AnnData objects representing the patches.
    - color_by: How to color the patches ('gene_expression' or 'total_expression').
    - gene_name: The name of the gene to use if color_by is 'gene_expression'.
    """
    # Check inputs
    if color_by == 'gene_expression' and gene_name is None:
        raise ValueError("You must specify a gene_name when color_by='gene_expression'.")

    # Collect all values for normalization
    values = []
    for adata_patch in adata_patches.values():
        if color_by == 'gene_expression':
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            values.append(expression)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            values.append(total_expression)
    
    # get min and max values
    values = np.array(values)
    min_value, max_value = values.min(), values.max()

    # Plot the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Plot each patch with the appropriate color
    for adata_patch in adata_patches.values():
        x_start, x_end, y_start, y_end = adata_patch.uns['patch_coords']
        
        if color_by == 'gene_expression':
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            normalized_value = (expression - min_value) / (max_value - min_value)
            color = plt.cm.viridis(normalized_value)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            normalized_value = (total_expression - min_value) / (max_value - min_value)
            color = plt.cm.viridis(normalized_value)
        
        # Draw a rectangle for the patch
        rect = mpatches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                  linewidth=1, edgecolor='none', facecolor=color, alpha=1)
        ax.add_patch(rect)

    # Create a color bar
    norm = plt.Normalize(min_value, max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(f'{gene_name} Expression' if color_by == 'gene_expression' else "total_expression")

    plt.axis('off')
    if is_pred:
        plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure2/' + data_type + '_' + gene_name + '_pred_rasterized.png', dpi=300, bbox_inches="tight")
    else:
        plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure2/' + data_type + '_' + gene_name + '_true_rasterized.png', dpi=300, bbox_inches="tight")
    plt.show()




import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import anndata

def rerasterize_patches(adata_patches, patch_size):
    """
    Adjusts the coordinates of patches to align them on a uniform grid.
    If two patches are merged into one, their expression values are averaged.
    
    Parameters:
    - adata_patches: Dictionary of AnnData objects representing the patches.
    - patch_size: The size of each patch.
    
    Returns:
    - new_adata_patches: Dictionary of AnnData objects with adjusted coordinates.
    """
    new_adata_patches = {}

    # Define grid step size
    step_size = patch_size

    # Create a set of unique x and y coordinates and sort them
    x_coords = sorted({adata_patch.uns['patch_coords'][0] for adata_patch in adata_patches.values()})
    y_coords = sorted({adata_patch.uns['patch_coords'][2] for adata_patch in adata_patches.values()})

    # Snap x and y coordinates to a uniform grid
    x_grid = np.arange(min(x_coords), max(x_coords) + step_size, step_size)
    y_grid = np.arange(min(y_coords), max(y_coords) + step_size, step_size)

    # A dictionary to keep track of merged patches and counts for averaging
    merged_patches = {}
    patch_counts = {}

    for idx, adata_patch in adata_patches.items():
        x_start, x_end, y_start, y_end = adata_patch.uns['patch_coords']
        
        # Find the nearest x and y positions on the grid
        x_center = (x_start + x_end) / 2
        y_center = (y_start + y_end) / 2

        new_x_start = x_grid[np.abs(x_grid - x_center).argmin()] - step_size / 2
        new_y_start = y_grid[np.abs(y_grid - y_center).argmin()] - step_size / 2
        new_x_end = new_x_start + patch_size
        new_y_end = new_y_start + patch_size

        # Create a unique key for the grid position
        grid_key = (new_x_start, new_y_start)

        if grid_key in merged_patches:
            # Merge the expression values by taking the mean
            existing_patch = merged_patches[grid_key]
            existing_patch.X = (existing_patch.X * patch_counts[grid_key] + adata_patch.X) / (patch_counts[grid_key] + 1)
            patch_counts[grid_key] += 1
        else:
            # Add the patch to the merged dictionary
            new_adata_patch = adata_patch.copy()
            new_adata_patch.uns['patch_coords'] = [new_x_start, new_x_end, new_y_start, new_y_end]
            merged_patches[grid_key] = new_adata_patch
            patch_counts[grid_key] = 1

    # Convert the merged patches to the final dictionary
    for idx, (grid_key, adata_patch) in enumerate(merged_patches.items()):
        new_adata_patches[idx] = adata_patch

    return new_adata_patches



# New resolution that doesn;t cause issues
new_resolution_xenium = 275

aligned_xenium_dictionary_rerastered = rerasterize_patches(aligned_xenium_dictionary, new_resolution_xenium)
aligned_visium_dictionary_rerastered = rerasterize_patches(aligned_visium_dictionary, new_resolution_xenium)

# combine the adata patches
combined_aligned_visium = combine_adata_patches(aligned_visium_dictionary_rerastered, adata_visium.uns['spatial'])
combined_aligned_xenium = combine_adata_patches(aligned_xenium_dictionary_rerastered, adata_visium.uns['spatial'])


len(aligned_xenium_dictionary_rerastered)


g = "ABCC11"

# plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary_rerastered, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium.uns['spatial'], aligned_visium_dictionary_rerastered, color_by='gene_expression', gene_name=g)

# save the data
# save the patches aligned_visium

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_patches_rerastered.pkl', 'wb') as f:
    pickle.dump(aligned_xenium_dictionary_rerastered, f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_rerastered.pkl', 'wb') as f:
    pickle.dump(aligned_visium_dictionary_rerastered, f)



# save all the data
combined_aligned_visium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data_rerastered.h5ad")
combined_aligned_xenium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_data_rerastered.h5ad")



############################################################################################################





# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


# combined data
adata_xenium_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
adata_visium_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_data.h5ad')

# make .X a csr matrix
adata_xenium_xenium.X = scipy.sparse.csr_matrix(adata_xenium_xenium.X)
adata_visium_xenium.X = scipy.sparse.csr_matrix(adata_visium_xenium.X)

# add array for gene expression
adata_xenium_xenium.X_array = pd.DataFrame(adata_xenium_xenium.X.toarray(), index=adata_xenium_xenium.obs.index)
adata_visium_xenium.X_array = pd.DataFrame(adata_visium_xenium.X.toarray(), index=adata_visium_xenium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_patches.pkl', 'rb') as f:
    aligned_visium_dictionary_xenium = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches.pkl', 'rb') as f:
    aligned_xenium_dictionary_xenium = pickle.load(f)

# plt.imshow(adata_visium.uns['spatial'])

# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary_xenium:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary_xenium[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary_xenium[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))


# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary_xenium:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary_xenium[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary_xenium[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# log transform the data
sc.pp.log1p(adata_xenium_xenium)
sc.pp.log1p(adata_visium_xenium)

# choose method
method = "visium"
# method = "xenium"

# img
# image_version = "visium"
image_version = "xenium"

# resolution based on image
resolution = 250

# edits
edit = "none"

# prepare the datasets
if method == "visium":
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary_xenium)
else:
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary_xenium)


# plot to confirm
plotRaster(adata_xenium_xenium.uns['spatial'], aligned_xenium_dictionary_xenium, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium_xenium.uns['spatial'], aligned_visium_dictionary_xenium, color_by='gene_expression', gene_name=g)


# New resolution that doesn;t cause issues
new_resolution_xenium = 275

aligned_xenium_dictionary_xenium_rerastered = rerasterize_patches(aligned_xenium_dictionary_xenium, new_resolution_xenium)
aligned_visium_dictionary_xenium_rerastered = rerasterize_patches(aligned_visium_dictionary_xenium, new_resolution_xenium)

len(aligned_xenium_dictionary_xenium_rerastered)


# plot the data
g = "TOMM7"


# plotRaster(adata_xenium_xenium.uns['spatial'], aligned_xenium_dictionary_xenium, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium_xenium.uns['spatial'], aligned_xenium_dictionary_xenium_rerastered, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium_xenium.uns['spatial'], aligned_visium_dictionary_xenium_rerastered, color_by='gene_expression', gene_name=g)



with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches_rerastered.pkl', 'wb') as f:
    pickle.dump(aligned_xenium_dictionary_xenium_rerastered, f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_patches_rerastered.pkl', 'wb') as f:
    pickle.dump(aligned_visium_dictionary_xenium_rerastered, f)










### NEed to transform the coordinates of the xenium data to match the visium data ###


# rotate the image and flip it, then make sure the values are correct
adata_xenium_reshaped = adata_xenium.copy()

# flip the image
adata_xenium_reshaped.uns['spatial'] = np.flip(adata_xenium_reshaped.uns['spatial'], axis=0)

# rotate the image
adata_xenium_reshaped.uns['spatial'] = np.rot90(adata_xenium_reshaped.uns['spatial'], k=3)



def transform_coordinates(x, y, image_width, image_height, rotation_k=0, flip_axis=None):
    """
    Transforms (x, y) coordinates to match image transformations using np.flip first, then np.rot90.
    
    Parameters:
    - x: Array of original x-coordinates.
    - y: Array of original y-coordinates.
    - image_width: Width of the image.
    - image_height: Height of the image.
    - rotation_k: Number of 90-degree rotations counterclockwise (0, 1, 2, or 3).
    - flip_axis: Axis to flip (None, 0 for vertical, 1 for horizontal).
    
    Returns:
    - x_new: Transformed x-coordinates (array of integers).
    - y_new: Transformed y-coordinates (array of integers).
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Step 1: Apply flipping using np.flip
    if flip_axis == 0:  # Vertical flip
        y = image_height - 1 - y
    elif flip_axis == 1:  # Horizontal flip
        x = image_width - 1 - x

    # Step 2: Apply rotation using np.rot90
    if rotation_k % 4 == 1:  # 90 degrees counterclockwise
        x_new = image_height - 1 - y
        y_new = x
    elif rotation_k % 4 == 2:  # 180 degrees
        x_new = image_width - 1 - x
        y_new = image_height - 1 - y
    elif rotation_k % 4 == 3:  # 270 degrees counterclockwise (90 degrees clockwise)
        x_new = y
        y_new = image_width - 1 - x
    else:  # rotation_k % 4 == 0, no rotation
        x_new, y_new = x, y

    # Ensure the final coordinates are integers
    x_new = np.round(x_new).astype(int)
    y_new = np.round(y_new).astype(int)

    return x_new, y_new



# Apply the transformation to x and y centroids
adata_xenium_reshaped.obs['x_centroid'], adata_xenium_reshaped.obs['y_centroid'] = transform_coordinates(
    adata_xenium_reshaped.obs['x_centroid'].to_numpy(),
    adata_xenium_reshaped.obs['y_centroid'].to_numpy(),
    adata_xenium_reshaped.uns['spatial'].shape[1],  # Width comes first for x-coordinates
    adata_xenium_reshaped.uns['spatial'].shape[0],  # Height comes second for y-coordinates
    rotation_k=1,
    flip_axis=0  # Vertical flip
)

# Update the 'spatial' coordinates in obsm
adata_xenium_reshaped.obsm['spatial'] = np.array([
    adata_xenium_reshaped.obs['x_centroid'],
    adata_xenium_reshaped.obs['y_centroid']
]).T


from utils import rasterizeGeneExpression_topatches

# rerasterize the data
aligned_xenium_dictionary_reshaped = rasterizeGeneExpression_topatches(adata_xenium_reshaped.uns['spatial'].astype(int), adata_xenium_reshaped, patch_size=resolution, visium=False)
aligned_xenium_dictionary_reshaped





# New resolution that doesn;t cause issues
new_resolution_xenium = 20

aligned_xenium_dictionary_rerastered = rerasterize_patches(aligned_xenium_dictionary_reshaped, new_resolution_xenium)

len(aligned_xenium_dictionary_rerastered)


# plot the data
g = "TOMM7"

# plotRaster(adata_xenium_reshaped.uns['spatial'], aligned_xenium_dictionary_reshaped, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium_reshaped.uns['spatial'], aligned_xenium_dictionary_rerastered, color_by='gene_expression', gene_name=g, is_pred=False)



