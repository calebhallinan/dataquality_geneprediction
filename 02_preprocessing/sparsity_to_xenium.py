
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


############################################################################################################

sparse_vals = [0, 1, 2, 3, 4, 5, 10, 15, 20]

for sparse_val in sparse_vals:

    print("sparse_val: ", sparse_val)

    # read in svg results
    gene_list = pd.read_csv("/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
    gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
    # these were not in the data
    gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


    ### read in aligned data ###

    # combined data
    adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
    adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_data.h5ad')

    # make .X a csr matrix
    adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
    adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

    # add array for gene expression
    adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
    adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

    # patches
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_patches.pkl', 'rb') as f:
        aligned_visium_dictionary = pickle.load(f)

    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches.pkl', 'rb') as f:
        aligned_xenium_dictionary = pickle.load(f)


    ############################################################################################################

    ax = adata_xenium.X.toarray()
    av = adata_visium.X.toarray()

    # iterate through the genes and look at the sparsity of the data, then save the data with less than 3 zeros
    ratios = []
    # save as an np array, so init with zeros
    ax_sparsity = np.zeros((len(ax), len(ax.T)))

    for i in range(len(ax.T)):

        # make xenium data 0 where visium data is 0
        # ax[:,i][av[:,i] == 0] = 0
        # make xenium data 0 where visium data is less than 3  
        ax[:,i][av[:,i] <= sparse_val] = 0

        ax_zeros = np.sum(ax[:,i] == 0)
        av_zeros = np.sum(av[:,i] == 0)

        # xenium divided by visiom
        ratio = ax_zeros/av_zeros
        ratios.append(ratio)
        # add to ax_sparsity
        ax_sparsity[:,i] = ax[:,i]

        # print("Gene:", i, "Xenium zeros:", ax_zeros, "Visium zeros:", av_zeros, "Ratio:", ratio)

    # add the sparsity to the data
    adata_xenium.X = scipy.sparse.csr_matrix(ax_sparsity)
    adata_xenium.X_array = ax_sparsity

    # split the data
    adata_patches_xenium = split_adata_patches(adata_xenium, aligned_xenium_dictionary)

    # change edit
    edit = "lessthanequalto" + str(sparse_val)

    # save all the data
    adata_xenium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sparsity/xenium_data_sparse_" + edit + ".h5ad")

    # save the patches aligned_xenium
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sparsity/xenium_patches_sparse_' + edit + '.pkl', 'wb') as f:
        pickle.dump(adata_patches_xenium, f)


############################################################################################################



