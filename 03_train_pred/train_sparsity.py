
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

# set gpu to 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

############################################################################################################

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]

# choose method
# method = "visium"
method = "xenium"

# img
# image_version = "visium"
image_version = "xenium"

# resolution
resolution = 250

# set seed
seed_val = 100
# Define the model
set_seed(seed_val)  # Set the seed for reproducibility



# read in rep2
if image_version == "visium":
    # combined data
    adata_visium_rep2 = sc.read_h5ad("/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/rep2_visiumimage_data.h5ad")

    # patches
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/rep2_visiumimage_patches.pkl', 'rb') as f:
        rep2_visium_dictionary = pickle.load(f)
    
    # analyze the results for full data
    X_train2, y_train2, scaled_coords2, correct_order2 = prepare_data(rep2_visium_dictionary)

else:
    # combined data
    adata_xenium_rep2 = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/rep2_xeniumimage_data.h5ad')

    # patches
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep2_aligned/rep2_xeniumimage_patches.pkl', 'rb') as f:
        rep2_xenium_dictionary = pickle.load(f)

    # analyze the results for full data
    X_train2, y_train2, scaled_coords2, correct_order2 = prepare_data(rep2_xenium_dictionary)


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

### read in aligned data ###

# sparse_valbda
# sparsity_values = [0, 1, 5, 10, 15, 20, 25, 50, 75, 100]
# sparsity_values = [5, 10, 15, 20]

sparse_val = 1

# for sparse_val in sparsity_values:

# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sparsity/xenium_data_sparse_lessthanequalto' + str(sparse_val) + '.h5ad')
# adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/visium_data_poissonsparse_valbda' + str(sparse_val) + '.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
# adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
# adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/visium_patches_poissonsparse_valbda' + str(sparse_val) + '.pkl', 'rb') as f:
#     aligned_visium_dictionary = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sparsity/xenium_patches_sparse_lessthanequalto' + str(sparse_val) + '.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)


# # scale the data and log transform
# scaling_factor = 1
# for i in aligned_visium_dictionary:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
#     # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# scale the data and log transform
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))

# log transform the data
sc.pp.log1p(adata_xenium)
# sc.pp.log1p(adata_visium)


# edits
edit = "sparse_lessthanequalto" + str(sparse_val)

# prepare the datasets
if method == "visium":
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary)
else:
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary)


# Prepare the data module
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

# Prepare the model
output_size = adata_xenium.shape[1]  # Assuming adata is defined
epochs = 150
model = GeneExpressionPredictor(output_size, dropout_rate=0.2, method = method, lossplot_save_file = f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_' + edit + '_lossplot.png')
print(model)

# Trainer initialization
trainer = pl.Trainer(max_epochs=epochs)

# Train the model
trainer.fit(model, data_module)

# Save the model based on resolution and file name
torch.save(model.state_dict(), f"/home/caleb/Desktop/improvedgenepred/results/sparsity/model_{method}data_image{image_version}_epochs{epochs}_seed{seed_val}_{edit}.pth")
# torch.save(model.state_dict(), f"/home/caleb/Desktop/improvedgenepred/results/models/modeltransformer_res{resolution}_epochs{epochs}_{method}data_seed{seed_val}_visiumimage_{edit}.pth")

# # Load the model
# model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/results/models/model_image{image_version}_epochs{epochs}_{method}data_seed{seed_val}_{edit}.pth"))

#############################################################################################################


# Evaluate the model
if method == "visium":
    correlation_df, adata_pred = evaluate_model_validation(
        data_module,
        model,
        adata_visium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_test_correlation_summary_{edit}.txt'
    )
else:
    correlation_df, adata_pred = evaluate_model_validation(
        data_module,
        model,
        adata_xenium,
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_test_correlation_summary_{edit}.txt'
    )

# adata_pred.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_test/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_aligned_adata_pred_full_' + edit + '.h5ad')
correlation_df.to_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_test_correlation_df_' + edit + '.csv')



# Prepare the data module
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
        output_file=f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_full_correlation_summary_{edit}.txt'
    )
else:
    correlation_df_full, adata_pred_full = evaluate_model_validation(
        data_module_full,
        model,
        adata_xenium,
        output_file=f'//home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_full_correlation_summary_{edit}.txt'
    )

# adata_pred_full.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_full/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_aligned_adata_pred_full_' + edit + '.h5ad')
correlation_df_full.to_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_full_correlation_df_' + edit + '.csv')



### REP 2 ###

if image_version == "visium":
    # evaluate the model
    correlation_df2, adata_pred2 = evaluate_model_validation(data_module_rep2, model, adata_visium_rep2, output_file=f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_rep2_correlation_summary_{edit}.txt')
else:
    # evaluate the model
    correlation_df2, adata_pred2 = evaluate_model_validation(data_module_rep2, model, adata_xenium_rep2, output_file=f'/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_{method}data_{image_version}image_seed{seed_val}_rep2_correlation_summary_{edit}.txt')

# adata_pred2.write('/home/caleb/Desktop/improvedgenepred/results/adata_pred_rep2/breastcancer_' + method + '_image' + image_version + '_seed' + str(seed_val) + '_adata_pred_' + edit + '.h5ad')
correlation_df2.to_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_rep2_correlation_df_' + edit + '.csv')


