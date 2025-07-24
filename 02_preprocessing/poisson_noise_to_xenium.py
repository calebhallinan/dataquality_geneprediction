
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


############################################################################################################

# lam_vals = [10, 15, 20, 25, 50, 75, 100]
lam_vals = [45]


for lam in lam_vals:

    print("lambda: ", lam)

    # read in svg results
    gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
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

    # add poisson noise

    # set seed
    np.random.seed(42)

    # lam = 5

    # iterate through each patch and add poisson noise to the gene expression
    for i in range(adata_xenium.shape[0]):
        adata_xenium.X[i] = scipy.sparse.csr_matrix(adata_xenium.X[i].toarray() + np.random.poisson(lam = lam, size = adata_xenium.X[i].shape[1]))


    # split the data
    adata_patches_xenium = split_adata_patches(adata_xenium, aligned_xenium_dictionary)


    # g = "HDC"
    # # plot the data to make sure it looks good
    # plotRaster(adata_xenium.uns['spatial'], adata_patches_xenium, color_by='gene_expression', gene_name=g)
    # plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)




    # save all the data
    adata_xenium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/xenium_data_poissonlambda"+ str(lam) + ".h5ad")


    # save the patches aligned_xenium
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/xenium_patches_poissonlambda'+ str(lam) + '.pkl', 'wb') as f:
        pickle.dump(adata_patches_xenium, f)



############################################################################################################


#### Add to visium data ####


# lam_vals = [10, 15, 20, 25, 50, 75, 100]
lam_vals = [5, 15, 45]


for lam in lam_vals:

    print("lambda: ", lam)

    # read in svg results
    gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
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



    # add poisson noise

    # set seed
    np.random.seed(42)

    # lam = 5

    # iterate through each patch and add poisson noise to the gene expression
    for i in range(adata_visium.shape[0]):
        adata_visium.X[i] = scipy.sparse.csr_matrix(adata_visium.X[i].toarray() + np.random.poisson(lam = lam, size = adata_visium.X[i].shape[1]))


    # split the data
    adata_patches_visium = split_adata_patches(adata_visium, aligned_visium_dictionary)


    # g = "HDC"
    # # plot the data to make sure it looks good
    # plotRaster(adata_xenium.uns['spatial'], adata_patches_xenium, color_by='gene_expression', gene_name=g)
    # plotRaster(adata_xenium.uns['spatial'], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)




    # save all the data
    adata_visium.write("/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/visium_data_poissonlambda"+ str(lam) + ".h5ad")


    # save the patches aligned_xenium
    with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/visium_patches_poissonlambda'+ str(lam) + '.pkl', 'wb') as f:
        pickle.dump(adata_patches_visium, f)



