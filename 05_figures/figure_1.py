

# Import libraries
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
import os
from PIL import Image
import matplotlib.patches as patches
import sys
sys.path.append('..')
from utils import *
import scipy
import pickle
import cv2



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
adata_visium.obsm['spatial'] = adata_visium.obsm['spatial'].astype(int)


# # get new cell centers for high rez image
# READ IN ALIGNED DATA
aligned_visium_points = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/aligned_visium_points_to_xenium_image.npy").astype(int)
adata_visium.obsm['spatial'] = aligned_visium_points

# subet gene list
adata_visium = adata_visium[:, gene_list]


### read in xenium data ### 


# file name
file_name = "breastcancer_xenium_sample1_rep1"
# resolution
resolution = 12
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

# add .obs
adata_xenium.obs = cell_centers
# add .obsm
adata_xenium.obsm["spatial"] = adata_xenium.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata_xenium.uns['spatial'] = img
# need to add this for subsetting
adata_xenium.obs.index = adata_xenium.obs.index.astype(str)

# subset the data
adata_xenium = adata_xenium[:, gene_list]

# make an array of the gene expression data
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)



############################################################################################################


###  Figure 1 ###


# plot visium spots and xenium spots on xenium image


import shapely
from shapely.geometry import MultiPoint
import matplotlib.patches as patches


Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
img = np.array(Image.open("/home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))

fig, ax = plt.subplots()

# Show your background image
ax.imshow(img)

# Plot Xenium points
x_xenium = adata_xenium.obsm["spatial"][:,0]
y_xenium = adata_xenium.obsm["spatial"][:,1]
ax.scatter(x_xenium, y_xenium, s=1, c="yellow", marker=".", edgecolor='none')

# Use Shapely to get a rotated bounding rectangle for Xenium points
points_xenium = MultiPoint(list(zip(x_xenium, y_xenium)))
rot_rect_xenium = points_xenium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords, y_coords = rot_rect_xenium.exterior.xy
polygon_xenium = patches.Polygon(
    xy=list(zip(x_coords, y_coords)),
    fill=False, edgecolor='C0', linewidth=2
)
ax.add_patch(polygon_xenium)

# # Plot Visium points
# x_visium = adata_visium.obsm["spatial"][:,0]
# y_visium = adata_visium.obsm["spatial"][:,1]
# ax.scatter(x_visium, y_visium, edgecolor="black", facecolors='none',
#            marker="o", linewidths=1, s=10)

# # Use Shapely to get a rotated bounding rectangle for Visium points
# points_visium = MultiPoint(list(zip(x_visium, y_visium)))
# rot_rect_visium = points_visium.minimum_rotated_rectangle

# # Convert Shapely polygon to a Matplotlib patch
# x_coords_v, y_coords_v = rot_rect_visium.exterior.xy
# polygon_visium = patches.Polygon(
#     xy=list(zip(x_coords_v, y_coords_v)),
#     fill=False, edgecolor='C1', linewidth=2
# )
# ax.add_patch(polygon_visium)

ax.axis("off")
# plt.show()
plt.savefig("/home/caleb/Desktop/improvedgenepred/05_figures/figure1/fig1_xenimage_xenpoints.png", dpi=300, bbox_inches='tight')
plt.close()

# read in visium image
img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium.npy")


fig, ax = plt.subplots()

# Show your background image
ax.imshow(img)

# # Plot Xenium points
# x_xenium = adata_xenium.obsm["spatial"][:,0]
# y_xenium = adata_xenium.obsm["spatial"][:,1]
# ax.scatter(x_xenium, y_xenium, s=1, c="yellow", marker=".", edgecolor='none')

# # Use Shapely to get a rotated bounding rectangle for Xenium points
# points_xenium = MultiPoint(list(zip(x_xenium, y_xenium)))
# rot_rect_xenium = points_xenium.minimum_rotated_rectangle

# # Convert Shapely polygon to a Matplotlib patch
# x_coords, y_coords = rot_rect_xenium.exterior.xy
# polygon_xenium = patches.Polygon(
#     xy=list(zip(x_coords, y_coords)),
#     fill=False, edgecolor='C0', linewidth=2
# )
# ax.add_patch(polygon_xenium)

# Plot Visium points
x_visium = adata_visium.obsm["spatial"][:,0]
y_visium = adata_visium.obsm["spatial"][:,1]
ax.scatter(x_visium, y_visium, edgecolor="black", facecolors='none',
           marker="o", linewidths=1, s=10)

# Use Shapely to get a rotated bounding rectangle for Visium points
points_visium = MultiPoint(list(zip(x_visium, y_visium)))
rot_rect_visium = points_visium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords_v, y_coords_v = rot_rect_visium.exterior.xy
polygon_visium = patches.Polygon(
    xy=list(zip(x_coords_v, y_coords_v)),
    fill=False, edgecolor='C1', linewidth=2
)
ax.add_patch(polygon_visium)

ax.axis("off")
# plt.show()
plt.savefig("/home/caleb/Desktop/improvedgenepred/05_figures/figure1/fig1_visimage_vispoints.png", dpi=300, bbox_inches='tight')
plt.close()




############################################################################################################

# create an image of a smiley face with color #D5BED9
from skimage.transform import resize

# Convert hex color to BGR for OpenCV
smiley_color_hex = "#EABED6"
smiley_color_rgb = tuple(int(smiley_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
smiley_color_bgr = (smiley_color_rgb[2], smiley_color_rgb[1], smiley_color_rgb[0])  # OpenCV uses BGR

# Create a blank white image (3 channels for color)
img = np.ones((100, 100, 3), dtype=np.uint8) * 255

# Draw the face (colored circle)
cv2.circle(img, (50, 50), 40, smiley_color_bgr, -1)  # Face

# Draw the eyes (white circles)
cv2.circle(img, (35, 40), 5, (255, 255, 255), -1)  # Left eye
cv2.circle(img, (65, 40), 5, (255, 255, 255), -1)  # Right eye

# Draw the smile (white ellipse)
cv2.ellipse(img, (50, 60), (20, 10), 0, 0, 180, (255, 255, 255), -1)  # Smile

# Save the image
cv2.imwrite('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face.png', img)


# now make the resolution worse
img = cv2.GaussianBlur(img, (31, 31), sigmaX=0)
# Save the image
cv2.imwrite('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_blurred.png', img)

# now rasterize the image into a 10x10 grid and plot

# Create a blank white image (1 channel for grayscale)
img = np.ones((100, 100), dtype=np.uint8) * 255

# Draw the face (black circle)
cv2.circle(img, (50, 50), 40, 0, -1)  # Face

# Draw the eyes (white circles)
cv2.circle(img, (35, 40), 5, 255, -1)  # Left eye
cv2.circle(img, (65, 40), 5, 255, -1)  # Right eye

# Draw the smile (white ellipse)
cv2.ellipse(img, (50, 60), (20, 10), 0, 0, 180, 255, -1)  # Smile


img_rasterized = resize(img, (10, 10), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.imshow(img_rasterized, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide the axes
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized.png', dpi=300, bbox_inches='tight')
# now plot on the rasterized image "gene expression" values: high where the black is, low where the white is

# Invert the rasterized image so black (0) becomes high (1), white (255) becomes low (0)
gene_expression_values = 1 - (img_rasterized.astype(float) / 255.0)

# Create a colormap
cmap = plt.cm.viridis

# Plot the gene expression values
plt.figure(figsize=(5, 5))
plt.imshow(gene_expression_values, cmap=cmap, interpolation='nearest')
plt.axis('off')  # Hide the axes
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized.png', dpi=300, bbox_inches='tight')


# now randomly select 10% of the pixels and set them to 0 to imitate sparsity
# set seed
np.random.seed(42)  # For reproducibility
sparsity_mask = np.random.rand(*img_rasterized.shape) < 0.1  # 10% sparsity
img_rasterized_sparse = img_rasterized.copy()
img_rasterized_sparse[sparsity_mask] = 255  # Set 10% of pixels to white (255)

# Plot the gene expression values for the sparse rasterized image
gene_expression_values_sparse = 1 - (img_rasterized_sparse.astype(float) / 255.0)
plt.figure(figsize=(5, 5))
plt.imshow(gene_expression_values_sparse, cmap=cmap, interpolation='nearest')
plt.axis('off')
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized_sparse_geneexp.png', dpi=300, bbox_inches='tight')


# now instead add noise to the rasterized image

img_rasterized_noisy = img_rasterized.copy()

# Add 100-200 values to every pixel at random
np.random.seed(42)
random_addition = np.random.randint(100, 250, size=img_rasterized_noisy.shape)
img_rasterized_noisy = np.clip(img_rasterized_noisy.astype(np.int32) - random_addition, 0, 255).astype(np.uint8)


img_rasterized_noisy = np.clip(
    img_rasterized_noisy.astype(np.int32), 0, 255
).astype(np.uint8)

# Plot the gene expression values for the noisy rasterized image
gene_expression_values_noisy = 1 - (img_rasterized_noisy.astype(float) / 255.0)
plt.figure(figsize=(5, 5))
plt.imshow(gene_expression_values_noisy, cmap=cmap, interpolation='nearest')
plt.axis('off')
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized_noisy_geneexp.png', dpi=300, bbox_inches='tight')


# now binarize the rasterized image such that you can see the smiley face only
img_rasterized_binary = (img_rasterized < 250).astype(np.uint8)

# Binarize the gene expression values so that the smiley face is perfectly visible
gene_expression_binary = (img_rasterized_binary == 0).astype(np.uint8)  # 1 where face, 0 elsewhere

plt.figure(figsize=(5, 5))
plt.imshow(gene_expression_binary, cmap=cmap, interpolation='nearest')
plt.axis('off')
# plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized_geneexp_binary.png', dpi=300, bbox_inches='tight')




img_rasterized2 = resize(img, (50, 50), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.imshow(img_rasterized2, cmap=cmap, interpolation='nearest')
plt.axis('off')  # Hide the axes


# Invert the rasterized image so black (0) becomes high (1), white (255) becomes low (0)
gene_expression_values2 = 1 - (img_rasterized2.astype(float) / 255.0)

# Create a colormap
cmap = plt.cm.viridis

# Plot the gene expression values
plt.figure(figsize=(5, 5))
plt.imshow(gene_expression_values2, cmap=cmap, interpolation='nearest')
plt.axis('off')  # Hide the axes
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/smiley_face_rasterized_imputed.png', dpi=300, bbox_inches='tight')


############################################################################################################

# plot a matrix of spatially correlated gene expression values of size 10x10


# Create a random 10x10 matrix
# set seed
np.random.seed(42)  # For reproducibility
random_matrix = np.random.rand(10, 10)

# Apply a Gaussian filter to introduce spatial correlation
spatially_correlated_gene_expression = gaussian_filter(random_matrix, sigma=2)

# Normalize to [0, 1]
spatially_correlated_gene_expression = (spatially_correlated_gene_expression - spatially_correlated_gene_expression.min()) / (spatially_correlated_gene_expression.max() - spatially_correlated_gene_expression.min())

# Create a colormap
cmap = plt.cm.viridis

# Plot the spatially correlated gene expression values
plt.figure(figsize=(5, 5))
plt.imshow(spatially_correlated_gene_expression, cmap=cmap, interpolation='nearest')
plt.axis('off')  # Hide the axes
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/spatially_correlated_gene_expression_matrix.png', dpi=300, bbox_inches='tight')


# Apply a Gaussian filter to introduce spatial correlation
spatially_correlated_gene_expression2 = gaussian_filter(random_matrix, sigma=1)

# Normalize to [0, 1]
spatially_correlated_gene_expression2 = (spatially_correlated_gene_expression2 - spatially_correlated_gene_expression2.min()) / (spatially_correlated_gene_expression2.max() - spatially_correlated_gene_expression2.min())

# Create a colormap
cmap = plt.cm.viridis

# Plot the spatially correlated gene expression values
plt.figure(figsize=(5, 5))
plt.imshow(spatially_correlated_gene_expression2, cmap=cmap, interpolation='nearest')
plt.axis('off')  # Hide the axes
plt.savefig('/home/caleb/Desktop/improvedgenepred/05_figures/figure1/spatially_correlated_gene_expression_matrix2.png', dpi=300, bbox_inches='tight')





############################################################################################################




# ##############################################################################################################


# file name
file_name = "breastcancer_xenium_sample1_rep2"
# resolution
resolution = 250
# read in the data
adata = sc.read_10x_h5('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/cell_feature_matrix.h5')

# Load the full-resolution spatial data
# cell_centers = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_visium_high_res_STalign.csv.gz", index_col=0)
# cell_centers = pd.read_csv("/home/caleb/Desktop/improvedgenepred/janesick_nature_comms_2023_companion/xenium_cell_centroids_visium_high_res.csv")
cell_centers = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_fullresolution_STalign.csv.gz", index_col=0)
# cell_centers = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/aligned_visium_points_to_xenium2_image.npy")
# cell_centers

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image"
img = np.array(Image.open("/home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))



# add .obs
adata.obs = cell_centers
# add .obsm
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata.uns['spatial'] = img
# need to add this for subsetting
adata.obs.index = adata.obs.index.astype(str)

# subset the data
adata = adata[:, gene_list]

# make an array of the gene expression data
adata.X_array = pd.DataFrame(adata.X.toarray(), index=adata.obs.index)

# need to subset bc there are negative values
adata = adata[adata.obs["y_centroid"] > 0]
adata = adata[adata.obs["x_centroid"] > 0]


# Extract patches using `extract_patches_from_centers` function
adata_patches_gb = rasterizeGeneExpression_topatches(adata.uns['spatial'], adata, patch_size=resolution, aggregation='sum')
len(adata_patches_gb)

# scale the data
scaling_factor = 1
for i in adata_patches_gb:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    adata_patches_gb[i].X = sc.pp.log1p(np.round(adata_patches_gb[i].X * scaling_factor))

# combine the adata patches
combined_adata_gb = combine_adata_patches(adata_patches_gb, adata.uns['spatial'])


# plot 
fig1 = plotRaster(adata.uns["spatial"], adata_patches_gb, color_by='total_expression')
fig1.savefig("/home/caleb/Desktop/improvedgenepred/05_figures/figure1/rep2_totalgeneexpression.svg", bbox_inches='tight', dpi=300)
plt.close(fig1)



def plotRaster(image, adata_patches, color_by='gene_expression', gene_name=None, if_vis=True):
    """
    Plots patches on the original image, colored by either gene expression or total expression.

    Parameters:
    - image: The original image array.
    - adata_patches: Dictionary of AnnData objects representing the patches.
    - color_by: How to color the patches ('gene_expression' or 'total_expression').
    - gene_name: The name of the gene to use if color_by is 'gene_expression'.
    - if_vis: Boolean flag to set the title as "Visium" (True) or "Xenium" (False).

    Returns:
    - fig: Matplotlib Figure object.
    """
    # Check inputs
    if color_by == 'gene_expression' and gene_name is None:
        raise ValueError("You must specify a gene_name when color_by='gene_expression'.")

    # Collect all values for normalization
    values = []
    for adata_patch in adata_patches.values():
        if color_by == 'gene_expression':
            # Sum the expression for the specified gene
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            values.append(expression)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            values.append(total_expression)

    values = np.array(values)
    min_value, max_value = values.min(), values.max()

    # Plot the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Plot each patch with the appropriate color
    for adata_patch in adata_patches.values():
        # Expecting patch_coords as (x_start, x_end, y_start, y_end)
        x_start, x_end, y_start, y_end = adata_patch.uns['patch_coords']

        if color_by == 'gene_expression':
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            normalized_value = (expression - min_value) / (max_value - min_value) if max_value > min_value else 0
            color = plt.cm.viridis(normalized_value)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            normalized_value = (total_expression - min_value) / (max_value - min_value) if max_value > min_value else 0
            color = plt.cm.viridis(normalized_value)

        # Draw a rectangle (square patch)
        rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                 linewidth=1, edgecolor='none', facecolor=color, alpha=1)
        ax.add_patch(rect)

    # Create a color bar at the bottom
    norm = plt.Normalize(min_value, max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label(f'{gene_name} Expression' if color_by == 'gene_expression' else "Total Expression")

    # Set the plot title
    title_prefix = "Visium" if if_vis else "Xenium"
    ax.set_title(f"{title_prefix} Expression of {gene_name}", fontsize=16)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


####################################################################################################################################

### Read in the Data ###


# file name
file_name = "breastcancer_xenium_sample1_rep1"
# resolution
resolution = 12
# read in the data
adata_xenium = sc.read_10x_h5('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/cell_feature_matrix.h5')

# Load the full-resolution spatial data
cell_centers = pd.read_csv(f"/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/{file_name}/{file_name}_fullresolution_STalign.csv.gz", index_col=0)
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/xenium_cell_centroids_visium_high_res.csv")
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/scaled_spots_for_xenium_image.csv", index_col=0)
# cell_centers.columns = ["x_centroid", "y_centroid"]

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
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

# adata_xenium.X = np.arcsinh(adata_xenium.X).toarray()

# scale genes with cpm
# sc.pp.normalize_total(adata_xenium, target_sum=1e6)

# log transform the data
# sc.pp.log1p(adata_xenium)

# sc.pp.normalize_total(adata_xenium, target_sum=1e6)

# get rid of genes that aren't in visium
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
# subset the data
adata_xenium = adata_xenium[:, gene_list]

# make an array of the gene expression data
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)

# plot
plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1], s=.01, c="black")
plt.axis("off")
plt.savefig("xenium1_spots.svg", dpi=300)


plt.figure(figsize=(18, 10), facecolor='black')
plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1], s=1, c="white")
plt.axis("off")
# plt.savefig("xenium1_spots.png", dpi=500, facecolor='black')


plt.figure(figsize=(18, 10), facecolor='black')
plt.scatter(
    adata_xenium.obsm["spatial"][:, 0], 
    adata_xenium.obsm["spatial"][:, 1], 
    s=1,  # Increase size for better visibility
    c="white",  # Fill color
    edgecolor='none',  # Remove edges
    alpha=1.0  # Set transparency to fully opaque
)
plt.axis("off")
plt.savefig("xenium1_spots.png", dpi=50, facecolor='black')



############################################################################################################

# plot visium spots and xenium spots on xenium image


import shapely
from shapely.geometry import MultiPoint
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches


Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
img = np.array(Image.open("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/" + file_name + "/" + img_name + ".tif"))

fig, ax = plt.subplots()

# Show your background image
ax.imshow(img)

# Plot Xenium points
x_xenium = adata_xenium.obsm["spatial"][:,0]
y_xenium = adata_xenium.obsm["spatial"][:,1]
ax.scatter(x_xenium, y_xenium, s=1, c="yellow", marker=".", edgecolor='none')

# Use Shapely to get a rotated bounding rectangle for Xenium points
points_xenium = MultiPoint(list(zip(x_xenium, y_xenium)))
rot_rect_xenium = points_xenium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords, y_coords = rot_rect_xenium.exterior.xy
polygon_xenium = patches.Polygon(
    xy=list(zip(x_coords, y_coords)),
    fill=False, edgecolor='C0', linewidth=2
)
ax.add_patch(polygon_xenium)

# Plot Visium points
x_visium = adata_visium.obsm["spatial"][:,0]
y_visium = adata_visium.obsm["spatial"][:,1]
ax.scatter(x_visium, y_visium, edgecolor="black", facecolors='none',
           marker="o", linewidths=1, s=10)

# Use Shapely to get a rotated bounding rectangle for Visium points
points_visium = MultiPoint(list(zip(x_visium, y_visium)))
rot_rect_visium = points_visium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords_v, y_coords_v = rot_rect_visium.exterior.xy
polygon_visium = patches.Polygon(
    xy=list(zip(x_coords_v, y_coords_v)),
    fill=False, edgecolor='C1', linewidth=2
)
ax.add_patch(polygon_visium)

ax.axis("off")
plt.show()

# read in visium image
img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium.npy")


fig, ax = plt.subplots()

# Show your background image
ax.imshow(img)

# Plot Xenium points
x_xenium = adata_xenium.obsm["spatial"][:,0]
y_xenium = adata_xenium.obsm["spatial"][:,1]
ax.scatter(x_xenium, y_xenium, s=1, c="yellow", marker=".", edgecolor='none')

# Use Shapely to get a rotated bounding rectangle for Xenium points
points_xenium = MultiPoint(list(zip(x_xenium, y_xenium)))
rot_rect_xenium = points_xenium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords, y_coords = rot_rect_xenium.exterior.xy
polygon_xenium = patches.Polygon(
    xy=list(zip(x_coords, y_coords)),
    fill=False, edgecolor='C0', linewidth=2
)
ax.add_patch(polygon_xenium)

# Plot Visium points
x_visium = adata_visium.obsm["spatial"][:,0]
y_visium = adata_visium.obsm["spatial"][:,1]
ax.scatter(x_visium, y_visium, edgecolor="black", facecolors='none',
           marker="o", linewidths=1, s=10)

# Use Shapely to get a rotated bounding rectangle for Visium points
points_visium = MultiPoint(list(zip(x_visium, y_visium)))
rot_rect_visium = points_visium.minimum_rotated_rectangle

# Convert Shapely polygon to a Matplotlib patch
x_coords_v, y_coords_v = rot_rect_visium.exterior.xy
polygon_visium = patches.Polygon(
    xy=list(zip(x_coords_v, y_coords_v)),
    fill=False, edgecolor='C1', linewidth=2
)
ax.add_patch(polygon_visium)

ax.axis("off")
plt.show()




############################################################################################################


### Read in the Data ###


# file name
file_name = "breastcancer_xenium_sample1_rep2"
# resolution
resolution = 250
# read in the data
adata = sc.read_10x_h5('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/cell_feature_matrix.h5')

# Load the full-resolution spatial data
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_visium_high_res_STalign.csv.gz", index_col=0)
# cell_centers = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/xenium_cell_centroids_visium_high_res.csv")
cell_centers = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep2/breastcancer_xenium_sample1_rep2_fullresolution_STalign.csv.gz", index_col=0)
cell_centers

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image"
img = np.array(Image.open("/home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))
# img = np.load("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/janesick_nature_comms_2023_companion/visium_high_res_image.npy")
plt.imshow(img)

# change image quality with gaussian blur
img = cv2.GaussianBlur(img,(5, 5), sigmaX=0)


# add .obs
adata.obs = cell_centers
# add .obsm
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata.uns['spatial'] = img
# need to add this for subsetting
adata.obs.index = adata.obs.index.astype(str)

# plot
plt.scatter(adata.obsm["spatial"][:,0], adata.obsm["spatial"][:,1], s=.01, c="black")
plt.axis("off")
# plt.savefig("xenium1_spots.svg", dpi=300)



############################################################################################################


### Gaussian Blur Example ###

plt.imshow(adata_xenium.uns['spatial'])

img = cv2.GaussianBlur(adata_xenium.uns['spatial'],(1001, 1001), sigmaX=0)

plt.imshow(img)
plt.axis("off")



############################################################################################################


# plot xenium image with xenium data and visium data

# file name
file_name = "breastcancer_xenium_sample1_rep1"
# resolution
resolution = 12
# read in the data
adata_xenium = sc.read_10x_h5('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/cell_feature_matrix.h5')

# Load the full-resolution spatial data
cell_centers = pd.read_csv(f"/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/{file_name}/{file_name}_fullresolution_STalign.csv.gz", index_col=0)

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
img = np.array(Image.open("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/" + file_name + "/" + img_name + ".tif"))


# plot
plt.figure(figsize=(18, 10))
plt.imshow(img)
plt.scatter(cell_centers["x_centroid"], cell_centers["y_centroid"], s=1, c="yellow", marker=".", edgecolor='none') # Remove edges)
plt.scatter(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1], s=25, edgecolor="black", facecolors='none', marker="o", linewidths=1)
plt.axis("off")
plt.savefig("fig1c_overlaps.png", dpi=1000)



# plot
plt.figure(figsize=(18, 10))
plt.imshow(img)
# plt.scatter(cell_centers["x_centroid"], cell_centers["y_centroid"], s=1, c="yellow", marker=".", edgecolor='none') # Remove edges)
# plt.scatter(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1], s=25, edgecolor="black", facecolors='none', marker="o", linewidths=1)
plt.scatter(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1], s=30, edgecolor="green", facecolors='none', marker="s", linewidths=1)
plt.axis("off")
plt.savefig("fig1c_rasterize.png", dpi=1000)



# for i in range(180,210):
#     print(i)
#     plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1], s=30, edgecolor="green", facecolors='none', marker="s", linewidths=1)
#     plt.scatter(adata_xenium.obsm["spatial"][i,0], adata_xenium.obsm["spatial"][i,1], s=30, edgecolor="red", facecolors='none', marker="s", linewidths=1)
#     plt.show()



# adata_visium.obsm["spatial"][42,0]
# adata_visium.obsm["spatial"][42,1]



# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
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

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)

# plt.imshow(adata_visium.uns['spatial'])

# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# adata_visium.X_array = adata_visium.X_array * scaling_factor

# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)



# for patch in aligned_xenium_dictionary:
#     if aligned_xenium_dictionary[patch].obs["x_centroid"][0].astype(int) == adata_xenium.obsm["spatial"][171,0] and aligned_xenium_dictionary[patch].obs["y_centroid"][0].astype(int) == adata_xenium.obsm["spatial"][171,1]:
#         print(patch)
#         break


# # iterate through the dictionary and find the most bottom right patch
# max_x = 0
# max_y = 0
# bottom_right_patch = None

# for patch in aligned_xenium_dictionary:
#     x_centroid = aligned_xenium_dictionary[patch].obs["x_centroid"][0].astype(int)
#     y_centroid = aligned_xenium_dictionary[patch].obs["y_centroid"][0].astype(int)
#     if x_centroid > max_x:
#         max_x = x_centroid
#         bottom_right_patch = patch
#     if y_centroid > max_y:
#         max_y = y_centroid
#         bottom_right_patch = patch

# print(f"Bottom right patch: {bottom_right_patch}, Coordinates: ({max_x}, {max_y})")



plt.imshow(aligned_xenium_dictionary["patch_1"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_patch1.png", dpi=300, bbox_inches="tight")



plt.imshow(aligned_xenium_dictionary["patch_2"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_patch2.png", dpi=300, bbox_inches="tight")


plt.imshow(aligned_xenium_dictionary["patch_3"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_patch3.png", dpi=300, bbox_inches="tight")



plt.imshow(aligned_xenium_dictionary["patch_4"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_patch4.png", dpi=300, bbox_inches="tight")


# plot the same patches for visium
plt.imshow(aligned_visium_dictionary["patch_1"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_visiumpatch1.png", dpi=300, bbox_inches="tight")

plt.imshow(aligned_visium_dictionary["patch_2"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_visiumpatch2.png", dpi=300, bbox_inches="tight")

plt.imshow(aligned_visium_dictionary["patch_3"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_visiumpatch3.png", dpi=300, bbox_inches="tight")

plt.imshow(aligned_visium_dictionary["patch_4"].uns["spatial"])
plt.axis("off")
plt.savefig("fig1c_visiumpatch4.png", dpi=300, bbox_inches="tight")



############################################################################################################


# Rerasterize the patches
aligned_visium_dictionary_pred = rerasterize_patches(aligned_visium_dictionary, 276)
aligned_xenium_dictionary_pred = rerasterize_patches(aligned_xenium_dictionary, 276)



# plot
plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary_pred, color_by='total_expression')
# plt.savefig("fig1c_xeniumtotalexpression.png", dpi=1000)
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary_pred, color_by='total_expression')
# plt.savefig("fig1c_visiumtotalexpression.png", dpi=1000)



############################################################################################################

# heatmap of gene expression

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_xenium_raw.h5ad')

sc.pp.log1p(adata_xenium)

sns.heatmap(adata_xenium.X.toarray(), cmap="bwr", cbar_kws={'label': 'Expression Level'}, cbar=False)
plt.axis("off")
plt.savefig("fig1d_geneheatmap.png", dpi=1000)



############################################################################################################



# SCVI gene expression imputation example


# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_xenium_raw_reshaped.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_visium_raw_reshaped.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_visium_dictionary_raw_reshaped.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw.pkl', 'rb') as f:
#     aligned_xenium_dictionary = pickle.load(f)
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw_reshaped.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)


# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# adata_visium.X_array = adata_visium.X_array * scaling_factor

# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))

# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)






# should be the name of image data in adata
tissue_section = "CytAssist_FFPE_Human_Breast_Cancer"

# file path where outs data is located
file_path = "/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_visium/"


### Read in adata ###

# read data
adata_visium2 = sc.read_visium(file_path)
# make unique
adata_visium2.var_names_make_unique()
# get mitochondrial gene expression info
adata_visium2.var["mt"] = adata_visium2.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata_visium2, qc_vars=["mt"], inplace=True)

# make spatial position str to integer
# https://discourse.scverse.org/t/data-fomr-new-spatial-transcriptomics-from-10x/1107/6
adata_visium2.obsm['spatial'] = adata_visium2.obsm['spatial'].astype(int)

# normalize data
# sc.pp.normalize_total(adata, inplace=True) # NOTE: same scale, proportion
# scp.normalize.library_size_normalize(adata)
# sc.pp.normalize_total(adata, target_sum=1e6)
# sc.pp.log1p(adata_visium)
# sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1000)

# scale data - CRUCIAL for doing when using the hires image to crop
adata_visium2.obsm['spatial'] = np.floor(adata_visium2.obsm["spatial"].astype(np.int64) * adata_visium2.uns['spatial'][tissue_section]["scalefactors"]["tissue_hires_scalef"]).astype(int)
adata_visium2.obsm['spatial'].shape


# Apply the transformation to x and y centroids
adata_visium2.obsm['spatial'] = np.array(transform_coordinates(
    adata_visium2.obsm['spatial'][:, 0],
    adata_visium2.obsm['spatial'][:, 1],
    adata_visium2.uns['spatial'][tissue_section]["images"]["hires"].shape[1],  # Width comes first for x-coordinates
    adata_visium2.uns['spatial'][tissue_section]["images"]["hires"].shape[0],  # Height comes second for y-coordinates
    rotation_k=1,
    flip_axis=0  # Vertical flip
)).T


# Create a mask based on whether each coordinate in visium_coords is in xenium_coords
def is_in_coords(coord, coords_array):
    return np.any(np.all(coords_array == coord, axis=1))

# Create a boolean mask for each coordinate in visium_coords
mask = np.array([is_in_coords(coord, np.array(adata_visium.obsm['spatial'])) for coord in np.array(adata_visium2.obsm['spatial'])])

# Use the mask to subset the AnnData object
adata_visium_in_overlap = adata_visium2[mask].copy()

# read in smoothed data
# smoothed_x = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_smoothed.csv", index_col=0)
# smoothed_x = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_smoothed_MAGIC.csv", index_col=0).T
# smoothed_x = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_imputed_SAVER.csv", index_col=0)
smoothed_x = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/visium_gene_expression_imputed_reshaped_SCVI.csv", index_col=0).T


# subset smoothed x
smoothed_x = smoothed_x.T[smoothed_x.columns.isin(list(adata_visium_in_overlap.obs.index))]

# make the index the same
adata_visium_in_overlap.obs.index = adata_visium.obs.index

# change uns
adata_visium_in_overlap.uns = adata_visium.uns

# replace
adata_visium_imputation = adata_visium_in_overlap.copy()

# change x to smoothed x
# get rid of these genes in spatial data
genes_to_remove = ['TKT', 'SLC39A4', 'GABARAPL2'] # NOTE: need to do for SCVI
adata_visium_imputation = adata_visium_imputation[:, [gene for gene in adata_visium_imputation.var_names if gene not in genes_to_remove]] # NOTE: need to do for SCVI
adata_visium_imputation.X = scipy.sparse.csr_matrix(smoothed_x)

# get just gene list
adata_visium_imputation = adata_visium_imputation[:, gene_list]

# log
sc.pp.log1p(adata_visium_imputation)



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
adata_patches = split_adata_patches(adata_visium_imputation, aligned_visium_dictionary)


g = "THAP2"
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_visium_imputation.uns['spatial'], adata_patches, color_by='gene_expression', gene_name=g)


# Rerasterize the patches
aligned_visium_dictionary_rerastered = rerasterize_patches(aligned_visium_dictionary, 17)
aligned_visium_dictionary_imputation_rerastered = rerasterize_patches(adata_patches, 17)



g = "THAP2"
plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary_rerastered, color_by='gene_expression', gene_name=g)
plt.savefig("fig1e_THAP2_normal.png", dpi=1000)
plotRaster(adata_visium_imputation.uns['spatial'], aligned_visium_dictionary_imputation_rerastered, color_by='gene_expression', gene_name=g)
plt.savefig("fig1e_THAP2_SCVI.png", dpi=1000)





############################################################################################################



# Sparsity example


# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_xenium_raw_reshaped.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_sample1_rep1_aligned_tovisiumimage/combined_aligned_visium_raw_reshaped.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_visium_dictionary_raw_reshaped.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw.pkl', 'rb') as f:
#     aligned_xenium_dictionary = pickle.load(f)
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/aligned_xenium_dictionary_raw_reshaped.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)


# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# adata_visium.X_array = adata_visium.X_array * scaling_factor

# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))

# log transform the data
# sc.pp.log1p(adata_xenium)
# sc.pp.log1p(adata_visium)




############################################################################################################


# schematic of scatterplot using random data

# generate random data that is normally distributed and correlated
n = 300
x = np.random.normal(0, 1, n)
y = x + np.random.normal(0, 1, n)

# plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=10, c="black")
plt.plot(x, x, c="red")
# plt.axis("off")
sns.despine()
# get rid of tick marks
plt.xticks([])
plt.yticks([])
# increase size of the border
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)


# schematic of histogram using random data

# generate random data that is normally distributed with different means
n = 300
x = np.random.normal(0, 1, n)
y = np.random.normal(1, 1, n)

# plot
plt.figure(figsize=(8, 5))
plt.hist(x, bins=30, color="#4D4FB0", alpha=.5)
plt.hist(y, bins=30, color="#D98F4B", alpha=.7)
# plt.axis("off")
sns.despine()
# get rid of tick marks
plt.xticks([])
plt.yticks([])
# increase size of the border
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
