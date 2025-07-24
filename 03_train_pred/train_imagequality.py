
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

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

############################################################################################################

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]

# image_version
image_version = "xenium"

# seed
seed_val = 42
# Define the model
set_seed(seed_val)  # Set the seed for reproducibility


### REad in REP2 data ###


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
cell_centers

# Load the full-resolution image
Image.MAX_IMAGE_PIXELS = None
img_name = "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image"
img = np.array(Image.open("/home/caleb/Desktop/improvedgenepred/data/" + file_name + "/" + img_name + ".tif"))
# img = np.load("/home/caleb/Desktop/improvedgenepred/janesick_nature_comms_2023_companion/visium_high_res_image.npy")
# img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium.npy")
# img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium2.npy")
# plt.imshow(img)

# add .obs
adata.obs = cell_centers
# add .obsm
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)
# add image
adata.uns['spatial'] = img
# need to add this for subsetting
adata.obs.index = adata.obs.index.astype(str)


# get rid of genes that aren't in visium
# subset the data
adata = adata[:, gene_list]

# make an array of the gene expression data
adata.X_array = pd.DataFrame(adata.X.toarray(), index=adata.obs.index)

# need to subset bc there are negative values
adata = adata[adata.obs["y_centroid"] > 0]
adata = adata[adata.obs["x_centroid"] > 0]


# plt.imshow(adata.uns['spatial'])



# Extract patches using `extract_patches_from_centers` function
adata_patches = rasterizeGeneExpression_topatches(adata.uns['spatial'], adata, patch_size=resolution, aggregation='sum')
len(adata_patches)

# scale the data
scaling_factor = 1
for i in adata_patches:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    adata_patches[i].X = sc.pp.log1p(np.round(adata_patches[i].X * scaling_factor))

# combine the adata patches
combined_adata = combine_adata_patches(adata_patches, adata.uns['spatial'])

# # Example call to plot patches based on a specific obs column
# plotRaster(adata.uns['spatial'], adata_patches, color_by='total_expression')
# plotRaster(adata.uns['spatial'], adata_patches, color_by='gene_expression', gene_name='HDC')



# analyze the results for full data
X_train2, y_train2, scaled_coords2, correct_order2 = prepare_data(adata_patches)

# Prepare the data module
data_module_rep2 = GeneExpressionDataModuleValidation(
    indices=correct_order2,
    coords=scaled_coords2,
    X_data=X_train2,
    y_data=y_train2,
    batch_size=64,
    val_pct=0,
    test_pct=1,
    seed=seed_val
)




### GO THROUGH THE DIFFERENT GB ###

# set the different gb
# gb_numbers = [5, 9, 15, 21, 25]
# gb_numbers = [9, 15,  21, 25]

# for gb in gb_numbers:

gb = 75

### read in aligned data ###

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_xenium_raw_gb' + str(gb) + '.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_visium_raw_gb' + str(gb) + '.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_visium_dictionary_raw_gb' + str(gb) + '.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_xenium_dictionary_raw_gb' + str(gb) + '.pkl', 'rb') as f:
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

# choose method
# method = "visium"
method = "xenium"

# resolution based on image
resolution = 250

# edits
edit = 'gb' + str(gb)

# prepare the datasets
if method == "visium":
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary)
else:
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary)


# plot the data
# g = "TOMM7"
# plot to confirm
# plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
# plotRaster(adata_visium.uns['spatial'], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)


############################################################################################################

# Define the data module
data_module = GeneExpressionDataModuleValidation(
    indices=correct_order,
    coords=scaled_coords,
    X_data=X_train,
    y_data=y_train,
    batch_size=64,
    val_pct=0.10,
    test_pct=0.15,
    seed=seed_val
)

# Define the model
output_size = adata_visium.shape[1]  # Assuming adata is defined
epochs = 150
model = GeneExpressionPredictor(output_size, dropout_rate=0.2, method = method, lossplot_save_file = f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_' + edit + '_lossplot.png')
print(model)

# Trainer initialization
trainer = pl.Trainer(max_epochs=epochs)

# Train the model
trainer.fit(model, data_module)

# Save the model based on resolution and file name
torch.save(model.state_dict(), f"/home/caleb/Desktop/improvedgenepred/results/img_quality/model_{method}data_{image_version}image_epochs{epochs}_seed{seed_val}_{edit}.pth")
# torch.save(model.state_dict(), f"/home/caleb/Desktop/improvedgenepred/results/models/modeltransformer_res{resolution}_epochs{epochs}_{method}datadata_seed{seed_val}_visiumimage_{edit}.pth")

# # Load the model
# model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/results/img_quality/model_{method}data_{image_version}image_epochs{epochs}_seed{seed_val}_{edit}.pth"))

#############################################################################################################


## Evaluate the Model ##


# Evaluate the model
if method == "visium":
    correlation_df, adata_pred = evaluate_model_validation(
        data_module,
        model,
        adata_visium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_test_correlation_summary_{edit}.txt'
    )
else:
    correlation_df, adata_pred = evaluate_model_validation(
        data_module,
        model,
        adata_xenium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_test_correlation_summary_{edit}.txt'
    )

# adata_pred.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_test/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_aligned_adata_pred_full_' + edit + '.h5ad')
correlation_df.to_csv('/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_test_correlation_df_' + edit + '.csv')



data_module_full = GeneExpressionDataModuleValidation(
    indices=correct_order,
    coords=scaled_coords,
    X_data=X_train,
    y_data=y_train,
    batch_size=64,
    val_pct=0,
    test_pct=1,
    seed=seed_val
)

# Evaluate the model
if method == "visium":
    correlation_df_full, adata_pred_full = evaluate_model_validation(
        data_module_full,
        model,
        adata_visium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_full_correlation_summary_{edit}.txt'
    )
else:
    correlation_df_full, adata_pred_full = evaluate_model_validation(
        data_module_full,
        model,
        adata_xenium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_full_correlation_summary_{edit}.txt'
    )

# adata_pred_full.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_full/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_aligned_adata_pred_full_' + edit + '.h5ad')
correlation_df_full.to_csv('/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_full_correlation_df_' + edit + '.csv')



## REP 2 ###


if method == "visium":
    correlation_df2, adata_pred2 = evaluate_model_validation(data_module_rep2, model, combined_adata, output_file = '/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_rep2_correlation_summary_' + edit + '.txt')
else:
    correlation_df2, adata_pred2 = evaluate_model_validation(data_module_rep2, model, combined_adata, output_file = '/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_rep2_correlation_summary_' + edit + '.txt')

# adata_pred2.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_rep2/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_adata_pred_' + edit + '.h5ad')
correlation_df2.to_csv('/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_rep2_correlation_df_' + edit + '.csv')



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
# img = np.load("/home/caleb/Desktop/improvedgenepred/janesick_nature_comms_2023_companion/visium_high_res_image.npy")
# img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium.npy")
# img = np.load("/home/caleb/Desktop/improvedgenepred/data/breastcancer_visium/spatial/tissue_alignedtoxenium2.npy")
# plt.imshow(img)

import cv2

# add gaussian blur
img = cv2.GaussianBlur(img, (gb, gb), 0)


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



# analyze the results for full data
X_train_gb, y_train_gb, scaled_coords_gb, correct_order_gb = prepare_data(adata_patches_gb)

# Prepare the data module
data_module_rep_gb = GeneExpressionDataModuleValidation(
    indices=correct_order_gb,
    coords=scaled_coords_gb,
    X_data=X_train_gb,
    y_data=y_train_gb,
    batch_size=64,
    val_pct=0,
    test_pct=1,
    seed=seed_val
)

    # Evaluate the model
if method == "visium":
    correlation_df_gb, adata_pred_gb = evaluate_model_validation(
        data_module_rep_gb,
        model,
        combined_adata_gb,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_rep2_blurred_correlation_summary_{edit}.txt'
    )
else:
    correlation_df_gb, adata_pred_gb = evaluate_model_validation(
        data_module_rep_gb,
        model,
        combined_adata_gb,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_{method}data_{image_version}image_seed{seed_val}_rep2_blurred_correlation_summary_{edit}.txt'
    )

# adata_pred_full.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_full/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_aligned_adata_pred_full_' + edit + '.h5ad')
correlation_df_gb.to_csv('/home/caleb/Desktop/improvedgenepred/results/img_quality/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_rep2_blurred_correlation_df_' + edit + '.csv')


# # perform shap analysis
# g = "LPL"
# data_module = GeneExpressionDataModule(correct_order, scaled_coords, X_train_tensor, y_train_tensor, batch_size=64, mode="test")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batch = next(iter(data_module.train_dataloader()))
# data, target, indices, coords = batch
# data = data.to(device)
# coords = coords.to(device)

# background = data[:59].to(device)
# test_images = data[59:64].to(device)
# test_images.shape

# background.shape

# e = shap.DeepExplainer(model.to(device), background)
# shap_values = e.shap_values(test_images, check_additivity=False)


# # Swap axes from [N, C, H, W] to [N, H, W, C] for visualization
# shap_numpy = [np.transpose(s, (1, 2, 0, 3)) for s in shap_values]
# test_numpy = np.transpose(test_images.detach().cpu().numpy(), (0, 2, 3, 1))

# # Undo transforms.ToTensor() by scaling values back to [0, 255]
# test_numpy = test_numpy * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
# test_numpy = (test_numpy * 255*255*255).astype(np.uint8)

# # g = "CTTN"
# # rerun for new gene
# shap_gene_numpy = np.array([s[..., gene_list.index(g)] for s in shap_numpy])  # Extract SHAP values for the gene
# # Plot the feature attributions for the selected gene
# shap.image_plot(shap_gene_numpy, test_numpy, show=False)
# plt.title(gene_list[gene_list.index(g)])





# ##############################################################################################################

