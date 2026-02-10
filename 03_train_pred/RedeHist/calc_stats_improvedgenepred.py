import scanpy as sc
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# import packages

############################################################################################################################

### functions used ###


# Function to subset and aggregate AnnData by bounding box coordinates
def subset_and_aggregate_patch_basedoncenters(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation='mean', visium=False):
    """Subset an AnnData object based on a spatial range and aggregate the data, ensuring cells are only included in the first patch they appear in."""
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    # filter spots within the bounding box and not already used
    mask = (
        (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) &
        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
    )
    
    # Remove cells that have already been used
    mask = mask & (~adata.obs.index.isin(used_cells))

    # Subset the AnnData object based on the mask
    adata_patch = adata[mask, :]

    # Return None if there are no cells in the patch
    if adata_patch.shape[0] == 0:
        return None

    # Add these cells to the set of used cells
    used_cells.update(adata_patch.obs.index)

    # Aggregate the data within the patch
    if aggregation == 'sum':
        aggregated_data = adata_patch.X.sum(axis=0)
    elif aggregation == 'mean':
        aggregated_data = adata_patch.X.mean(axis=0)
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")

    # Create a new AnnData object with aggregated data
    aggregated_data = aggregated_data if isinstance(aggregated_data, csr_matrix) else csr_matrix(aggregated_data)
    new_adata = ad.AnnData(X=aggregated_data)
    
    # Add image patch
    new_adata.uns['spatial'] = image[y_start:y_end, x_start:x_end]
    # Add patch coordinates
    new_adata.uns['patch_coords'] = [x_start, x_end, y_start, y_end]
    
    # Add centroid of new patch
    new_adata.obs['x_centroid'] = (x_start + x_end) / 2
    new_adata.obs['y_centroid'] = (y_start + y_end) / 2

    # if visium:
    #     for field in ['in_tissue', 'array_row', 'array_col']:
    #         new_adata.obs[field] = adata_patch.obs[field].iloc[0]

    # Add spatial coordinates
    new_adata.obsm["spatial"] = new_adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)

    # Add variables and gene names
    new_adata.var = adata.var
    new_adata.var_names = adata.var_names
    # make sure X is a sparse matrix
    new_adata.X = csr_matrix(new_adata.X)

    return new_adata

# Function to extract patches and aggregate data from an image and AnnData object based on supplied center coordinates
def rasterizeGeneExpression_topatches_basedoncenters(image, adata, center_coords, patch_size=100, aggregation='mean', visium=False):
    """Extract patches centered around supplied coordinates from an image and aggregate AnnData data accordingly."""

    # Initialize variables
    adata_sub_dict = {}
    img_height, img_width, _ = image.shape
    used_cells = set()

    # Loop through each center coordinate
    for patch_index, (x_center, y_center) in enumerate(center_coords):
        # Calculate bounding box around the center coordinate
        x_start = max(0, x_center - patch_size // 2)
        x_end = min(img_width, x_center + patch_size // 2)
        y_start = max(0, y_center - patch_size // 2)
        y_end = min(img_height, y_center + patch_size // 2)

        # Subset and aggregate the AnnData object
        adata_patch = subset_and_aggregate_patch_basedoncenters(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation, visium)
        
        # Filter out empty patches
        if adata_patch is not None:
            if adata_patch.uns['spatial'].shape == (patch_size, patch_size, 3):
                patch_name = f"patch_{patch_index}"
                adata_sub_dict[patch_name] = adata_patch

    # return the dictionary of patches
    return adata_sub_dict




import matplotlib.patches as mpatches
# Function to plot patches on the original image
def plotRaster(image, adata_patches, color_by='gene_expression', gene_name=None):
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
    plt.show()




def combine_adata_patches(adata_patches, image):
    # Initialize list to collect data
    adata_list = []

    # Iterate over the dictionary to prepare the data for concatenation
    for key, adata in adata_patches.items():
        # Set the index for each observation to the dictionary key
        adata.obs.index = [key] * adata.shape[0]
        adata.var_names_make_unique()
        adata_list.append(adata)
        print([key] * adata.shape[0])

    # Concatenate all the adata objects
    combined_adata = ad.concat(adata_list, merge='same', uns_merge='same')
    # add image
    combined_adata.uns['spatial'] = image
    # add X_array
    combined_adata.X_array = pd.DataFrame(combined_adata.X.toarray(), index=combined_adata.obs.index)

    return combined_adata




# Function to subset and aggregate AnnData by bounding box coordinates
def subset_and_aggregate_patch(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation='mean',visium=False):
    """Subset an AnnData object based on a spatial range and aggregate the data, ensuring cells are only included in the first patch they appear in."""
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    # filter spots within the bounding box and not already used
    mask = (
        (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) &
        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
    )
    
    # Remove cells that have already been used
    mask = mask & (~adata.obs.index.isin(used_cells))

    # Subset the AnnData object based on the mask
    adata_patch = adata[mask, :]

    # Return None if there are no cells in the patch
    if adata_patch.shape[0] == 0:
        return None

    # Add these cells to the set of used cells
    used_cells.update(adata_patch.obs.index)

    # Aggregate the data within the patch
    if aggregation == 'sum':
        aggregated_data = adata_patch.X.sum(axis=0)
    elif aggregation == 'mean':
        aggregated_data = adata_patch.X.mean(axis=0)
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")

    # Create a new AnnData object with aggregated data
    aggregated_data = aggregated_data if isinstance(aggregated_data, csr_matrix) else csr_matrix(aggregated_data)
    new_adata = ad.AnnData(X=aggregated_data)
    
    # Add image patch
    new_adata.uns['spatial'] = image[y_start:y_end, x_start:x_end]
    # Add patch coordinates
    new_adata.uns['patch_coords'] = [x_start, x_end, y_start, y_end]
    
    # Add centroid of new patch
    new_adata.obs['x_centroid'] = (x_start + x_end) / 2
    new_adata.obs['y_centroid'] = (y_start + y_end) / 2

    # Aggregate and sum specific fields
    # for field in ['transcript_counts', 'control_probe_counts', 'control_codeword_counts', 'total_counts', 'cell_area', 'nucleus_area']:
    #     new_adata.obs[field] = adata_patch.obs[field].sum()

    if visium:
        for field in ['in_tissue', 'array_row', 'array_col']:
            new_adata.obs[field] = adata_patch.obs[field].iloc[0]

    # Add spatial coordinates
    new_adata.obsm["spatial"] = new_adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)

    # Add variables and gene names
    new_adata.var = adata.var
    new_adata.var_names = adata.var_names

    return new_adata

# Function to extract patches and aggregate data from an image and AnnData object
def rasterizeGeneExpression_topatches(image, adata, patch_size=100, aggregation='mean', visium=False):
    """Extract non-overlapping patches from an image and aggregate AnnData data accordingly."""

    # Initialize variables
    adata_sub_dict = {}
    img_height, img_width, _ = image.shape
    used_cells = set()

    if visium:
        # Determine the bounding box using the centroids - same method as SEraster
        x_min = (np.floor(adata.obsm['spatial'][:,0].min()) - patch_size/2).astype(int)
        x_max = (np.ceil(adata.obsm['spatial'][:,0].max()) + patch_size/2).astype(int)
        y_min = (np.floor(adata.obsm['spatial'][:,1].min()) - patch_size/2).astype(int)
        y_max = (np.ceil(adata.obsm['spatial'][:,1].max()) + patch_size/2).astype(int)
    else:
        # Determine the bounding box using the centroids - same method as SEraster
        x_min = (np.floor(adata.obsm['spatial'].min()) - patch_size/2).astype(int)
        x_max = (np.ceil(adata.obsm['spatial'].max()) + patch_size/2).astype(int)
        y_min = (np.floor(adata.obsm['spatial'].min()) - patch_size/2).astype(int)
        y_max = (np.ceil(adata.obsm['spatial'].max()) + patch_size/2).astype(int)

    # init start points
    x_starts = np.arange(x_min, x_max, patch_size)
    y_starts = np.arange(y_min, y_max, patch_size)

    # get total patches
    total_patches = len(x_starts) * len(y_starts)
    patch_index = 0

    # Loop through all patches
    for y_start in y_starts:
        for x_start in x_starts:
            x_end = min(x_start + patch_size, img_width)
            y_end = min(y_start + patch_size, img_height)

            # Subset and aggregate the AnnData object
            adata_patch = subset_and_aggregate_patch(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation, visium)
            
            # Filter out empty patches
            if adata_patch is not None:
                if adata_patch.uns['spatial'].shape == (patch_size, patch_size, 3):
                    patch_name = f"patch_{patch_index}"
                    adata_sub_dict[patch_name] = adata_patch

            patch_index += 1

    # return the dictionary of patches
    return adata_sub_dict

############################################################################################################################


# read in adata objects
adata_vis_pred = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/07_revision/RedeHist/example_Visium/HBC_Visium.predict.h5ad')
adata_vis_true = sc.read_visium('/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/')

# reorder this
adata_vis_true.obsm['spatial'] = adata_vis_true.obsm['spatial'].astype(int)*adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']["scalefactors"]["tissue_hires_scalef"]



############################################################################################################################



### read in xenium ###
from PIL import Image
import pandas as pd

# read in adata objects
adata_xen_pred = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/07_revision/RedeHist/example_Xenium/HBC_Xenium.predict.h5ad')


# file name
file_name = "breastcancer_xenium_sample1_rep1"
# resolution
resolution = 250
# read in the data
adata_xenium = sc.read_10x_h5('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/cell_feature_matrix.h5')

# Load the full-resolution spatial data
cell_centers = pd.read_csv(f"/home/caleb/Desktop/improvedgenepred/data/{file_name}/{file_name}_fullresolution_STalign.csv.gz", index_col=0)

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image"
img = np.array(Image.open("//home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))
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

# NO NORMALIZATION HERE

# # get rid of genes that aren't in visium
# gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
# gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
# # subset the data
# adata_xenium = adata_xenium[:, gene_list]

# make an array of the gene expression data
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)

# plt.imshow(adata_xenium.uns['spatial'])

# plot the data
plt.figure(figsize=(18, 10))
plt.imshow(adata_xenium.uns['spatial'])
plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1], s=1, c="yellow")
plt.axis("off")





import numpy as np
import matplotlib.pyplot as plt

img = adata_xenium.uns['spatial']
h, w = img.shape[:2]

x = adata_xen_pred.obsm["spatial"][:, 0]
y = adata_xen_pred.obsm["spatial"][:, 1]

# Undo fliplr (on rotated image)
x1 = h - 1 - x
y1 = y

# Undo rot90(k=-1) — CORRECT axis + offset
x_orig = y1
y_orig = x1

# x_orig, y_orig are your current (almost-correct) coordinates

# Flip vertically in ORIGINAL image space
y_final = h - 1 - y_orig
x_final = x_orig


# Now x_orig, y_orig align with ORIGINAL image
plt.figure(figsize=(18, 10))
plt.imshow(img)
plt.scatter(x_final, y_final, s=1, c="yellow")
plt.axis("off")



# stack spatial coords
adata_xen_pred.obsm["spatial"] = np.column_stack([x_final, y_final])


# rasterize
rast_xen_true = rasterizeGeneExpression_topatches(adata_xenium.uns['spatial'], adata_xenium, patch_size=250, aggregation='sum', visium=False)

rast_xen_true_data = combine_adata_patches(rast_xen_true, adata_xenium.uns['spatial'])

# plotRaster(adata_xenium.uns['spatial'], rast_xen_true, color_by='gene_expression', gene_name='CEACAM6')


rast_xen_pred = rasterizeGeneExpression_topatches(adata_xenium.uns['spatial'], adata_xen_pred, patch_size=250, aggregation='sum', visium=False)

# rast_xen_pred_data = combine_adata_patches(rast_xen_pred, adata_xenium.uns['spatial'])

# plotRaster(adata_xenium.uns['spatial'], rast_xen_pred, color_by='gene_expression', gene_name='CEACAM6')


# rasterize based on centers
rast_xen_pred_basedoncenters = rasterizeGeneExpression_topatches_basedoncenters(adata_xenium.uns['spatial'], adata_xen_pred, rast_xen_true_data.obsm['spatial'], patch_size=250, aggregation='sum', visium=False)


# plotRaster(adata_xenium.uns['spatial'], rast_xen_pred_basedoncenters, color_by='gene_expression', gene_name='CEACAM6')


rast_xen_pred_data = combine_adata_patches(rast_xen_pred_basedoncenters, adata_xenium.uns['spatial'])



# Subset to same 306 genes and find pearson

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
len(gene_list)

# NOTE: need to remove a few more to for some reason

gene_list = [gene for gene in gene_list if gene not in ['IL3RA', 'KRT6B', 'SCGB2A1', 'TCIM', 'TENT5C']]
len(gene_list)

# subset the data
rast_xen_true_data = rast_xen_true_data[:, gene_list]
rast_xen_pred_data = rast_xen_pred_data[:, gene_list]



import numpy as np
import pandas as pd

# Align two AnnData objects by identical spatial coordinates
def align_adata_by_spatial(adata_true, adata_pred, decimals=6):
    def spatial_keys(adata):
        return pd.Series(
            [f"{x:.{decimals}f}_{y:.{decimals}f}" for x, y in adata.obsm["spatial"]],
            index=adata.obs_names,
        )

    keys_true = spatial_keys(adata_true)
    keys_pred = spatial_keys(adata_pred)

    shared_keys = np.intersect1d(keys_true.values, keys_pred.values)

    adata_true_aligned = adata_true[keys_true.isin(shared_keys)].copy()
    adata_pred_aligned = adata_pred[keys_pred.isin(shared_keys)].copy()

    # Ensure identical ordering
    order = (
        pd.Index(
            [f"{x:.{decimals}f}_{y:.{decimals}f}" 
             for x, y in adata_true_aligned.obsm["spatial"]]
        )
        .get_indexer(
            [f"{x:.{decimals}f}_{y:.{decimals}f}" 
             for x, y in adata_pred_aligned.obsm["spatial"]]
        )
    )

    adata_true_aligned = adata_true_aligned[order].copy()

    return adata_true_aligned, adata_pred_aligned


# align
rast_xen_true_aligned, rast_xen_pred_aligned = align_adata_by_spatial(
    rast_xen_true_data,
    rast_xen_pred_data
)

# Sanity check
assert np.allclose(
    rast_xen_true_aligned.obsm["spatial"],
    rast_xen_pred_aligned.obsm["spatial"]
)





############################################################################################################################


# get image
img = adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['images']['hires']


# plot original
plt.figure(figsize=(18, 10))
plt.imshow(img)
plt.scatter(adata_vis_true.obsm['spatial'][:,0], adata_vis_true.obsm['spatial'][:,1], s=1, c="yellow")
plt.axis("off")




import numpy as np
import matplotlib.pyplot as plt

img = adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['images']['hires']
h, w = img.shape[:2]

x = adata_vis_pred.obsm["spatial"][:, 0]*adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']["scalefactors"]["tissue_hires_scalef"]
y = adata_vis_pred.obsm["spatial"][:, 1]*adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']["scalefactors"]["tissue_hires_scalef"]

# Undo fliplr (on rotated image)
x1 = h - 1 - x
y1 = y

# Undo rot90(k=-1) — CORRECT axis + offset
x_orig = y1
y_orig = x1

# x_orig, y_orig are your current (almost-correct) coordinates

# Flip vertically in ORIGINAL image space
y_final = h - 1 - y_orig
x_final = x_orig


# ✅ Now x_orig, y_orig align with ORIGINAL image
plt.figure(figsize=(18, 10))
plt.imshow(img)
plt.scatter(x_final, y_final, s=1, c="yellow")
plt.axis("off")


# stack
adata_vis_pred.obsm["spatial"] = np.column_stack([x_final, y_final])




## Figure out spacing of visium ##

import numpy as np
from scipy.spatial import cKDTree

# Paired (x, y) coordinates
coords = adata_vis_true.obsm["spatial"].astype(int)

# Build KD-tree for fast neighbor lookup
tree = cKDTree(coords)

# For each point, find its nearest neighbor (excluding itself)
distances, _ = tree.query(coords, k=2)
nearest_dist = distances[:, 1]  # distance to closest distinct point

# Grid spacing = most common nearest-neighbor distance
spacing = int(np.round(np.median(nearest_dist)))

spacing




# adata_vis_true.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']["scalefactors"]["tissue_hires_scalef"]*250

# rasterize
rast_vis_true = rasterizeGeneExpression_topatches_basedoncenters(img, adata_vis_true, adata_vis_true.obsm['spatial'].astype(int), patch_size=16, aggregation='sum', visium=True)
len(rast_vis_true)


rast_vis_true_data = combine_adata_patches(rast_vis_true, img)


# plotRaster(img, rast_vis_true, color_by='gene_expression', gene_name='CEACAM6')


rast_vis_pred = rasterizeGeneExpression_topatches_basedoncenters(img, adata_vis_pred, adata_vis_true.obsm['spatial'].astype(int), patch_size=16, aggregation='sum', visium=False)
len(rast_vis_pred)

rast_vis_pred_data = combine_adata_patches(rast_vis_pred, img)

# plotRaster(img, rast_vis_pred, color_by='gene_expression', gene_name='CEACAM6')





# Subset to same 306 genes and find pearson

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]
len(gene_list)

# NOTE: need to remove a few more to for some reason

gene_list = [gene for gene in gene_list if gene not in ['IL3RA', 'KRT6B', 'SCGB2A1', 'TCIM', 'TENT5C']]
len(gene_list)

# subset the data
rast_vis_true_data = rast_vis_true_data[:, gene_list]
rast_vis_pred_data = rast_vis_pred_data[:, gene_list]


rast_vis_true_data.var
rast_vis_pred_data.var


# Usage
rast_vis_true_aligned, rast_vis_pred_aligned = align_adata_by_spatial(
    rast_vis_true_data,
    rast_vis_pred_data
)

# Sanity check
assert np.allclose(
    rast_xen_true_aligned.obsm["spatial"],
    rast_xen_pred_aligned.obsm["spatial"]
)



### plot both datasets ###

## plot pearson and RMSE ###
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# compute gene-wise Pearson correlations and RMSE 
# Xenium data
X_true_xen = rast_xen_true_aligned.X
X_pred_xen = rast_xen_pred_aligned.X

# handle sparse matrices
if hasattr(X_true_xen, "toarray"):
    X_true_xen = X_true_xen.toarray()
if hasattr(X_pred_xen, "toarray"):
    X_pred_xen = X_pred_xen.toarray()

gene_corrs_xen = []
gene_rmse_xen = []

for i in range(X_true_xen.shape[1]):
    # Pearson correlation
    gene_corrs_xen.append(pearsonr(X_true_xen[:, i], X_pred_xen[:, i])[0])
    # RMSE normalized by range
    range_xen = X_true_xen[:, i].max() - X_true_xen[:, i].min()
    rmse_xen = mean_squared_error(X_true_xen[:, i], X_pred_xen[:, i], squared=False)
    gene_rmse_xen.append(rmse_xen / range_xen if range_xen != 0 else np.nan)

# wrap in DataFrame
xenium_xenimage_corr = pd.DataFrame({"Pearson": gene_corrs_xen, "Normalized_RMSE": gene_rmse_xen})

# Visium data
X_true_vis = rast_vis_true_aligned.X
X_pred_vis = rast_vis_pred_aligned.X

# handle sparse matrices
if hasattr(X_true_vis, "toarray"):
    X_true_vis = X_true_vis.toarray()
if hasattr(X_pred_vis, "toarray"):
    X_pred_vis = X_pred_vis.toarray()

gene_corrs_vis = []
gene_rmse_vis = []

for i in range(X_true_vis.shape[1]):
    # Pearson correlation
    gene_corrs_vis.append(pearsonr(X_true_vis[:, i], X_pred_vis[:, i])[0])
    # RMSE normalized by range
    range_vis = X_true_vis[:, i].max() - X_true_vis[:, i].min()
    rmse_vis = mean_squared_error(X_true_vis[:, i], X_pred_vis[:, i], squared=False)
    gene_rmse_vis.append(rmse_vis / range_vis if range_vis != 0 else np.nan)

# wrap in DataFrame
visium_visimage_corr = pd.DataFrame({"Pearson": gene_corrs_vis, "Normalized_RMSE": gene_rmse_vis})



#  plot correlation distributions 
plt.figure(figsize=(10, 5))


# Visium
sns.histplot(
    visium_visimage_corr["Pearson"],
    color="#56B4E9",
    label="Visium data - Visium Image",
    kde=True
)

# plot average correlation for Visium
visium_mean = np.mean(visium_visimage_corr["Pearson"])
plt.axvline(visium_mean, color="#56B4E9", linestyle="--")

# annotate mean for Visium
plt.text(
    visium_mean,
    plt.ylim()[1] * 0.8,
    f"{np.round(visium_mean, 3)}",
    fontsize=10,
    color="#56B4E9",
    ha="center"
)

# Xenium
sns.histplot(
    xenium_xenimage_corr["Pearson"],
    color="#E69F01",
    label="Xenium data - Xenium Image",
    kde=True
)

# plot average correlation for Xenium
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# annotate mean for Xenium
plt.text(
    xenium_mean,
    plt.ylim()[1] * 0.9,
    f"{np.round(xenium_mean, 3)}",
    fontsize=10,
    color="#E69F01",
    ha="center"
)


plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig13_histogram.svg", dpi=1000, bbox_inches="tight")



### RMSE ###


# plot correlation distributions 
plt.figure(figsize=(10, 5))


# Visium
sns.histplot(
    visium_visimage_corr["Normalized_RMSE"],
    color="#56B4E9",
    label="Visium data - Visium Image",
    kde=True
)

# plot average correlation for Visium
visium_mean = np.mean(visium_visimage_corr["Normalized_RMSE"])
plt.axvline(visium_mean, color="#56B4E9", linestyle="--")

# annotate mean for Visium
plt.text(
    visium_mean,
    plt.ylim()[1] * 0.8,
    f"{np.round(visium_mean, 3)}",
    fontsize=10,
    color="#56B4E9",
    ha="center"
)

# Xenium
sns.histplot(
    xenium_xenimage_corr["Normalized_RMSE"],
    color="#E69F01",
    label="Xenium data - Xenium Image",
    kde=True
)

# plot average correlation for Xenium
xenium_mean = np.mean(xenium_xenimage_corr["Normalized_RMSE"])
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# annotate mean for Xenium
plt.text(
    xenium_mean,
    plt.ylim()[1] * 0.9,
    f"{np.round(xenium_mean, 3)}",
    fontsize=10,
    color="#E69F01",
    ha="center"
)

plt.xlabel("Normalized_RMSE")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig13_histogram2.svg", dpi=1000, bbox_inches="tight")






### scatterplots ###

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["Pearson"], y=xenium_xenimage_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Pearson Correlation Values")
plt.xlim(0, 1)
plt.ylim(0, 1)
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig13_scatterplot1.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["Normalized_RMSE"], y=xenium_xenimage_corr["Normalized_RMSE"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1.3], [0, 1.3], color="gray", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Normalized_RMSE")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig13_scatterplot2.svg", dpi=1000, bbox_inches="tight")






############################################################################################################################

