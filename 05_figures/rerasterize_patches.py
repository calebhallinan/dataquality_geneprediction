
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from PIL import Image
from adjustText import adjust_text
import scanpy as sc
import scipy
import pickle
import sys
sys.path.append('..')
from plotting_utils import *
import pickle
import scanpy as sc
import scipy.sparse


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


# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))


# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)


##########################################################################################################################################



# read in adata
adata_visium_pred = sc.read_h5ad("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_adata_pred_full_none.h5ad")
adata_xenium_pred = sc.read_h5ad("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_adata_pred_full_none.h5ad")


import tqdm

def split_adata_by_key(adata, aligned_dict):
    """
    Splits an AnnData object into smaller AnnData objects based on unique obs index values.

    Parameters:
    -----------
    adata : AnnData
        The input AnnData object to be split.
    aligned_dict : dict
        A dictionary containing alignment information for each patch key.

    Returns:
    --------
    dict
        A dictionary where each key is a unique obs index, and the value is the corresponding split AnnData object.
    """
    # Initialize the dictionary to store split AnnData objects
    adata_patches = {}
    
    # Extract unique keys from the obs index
    unique_keys = adata.obs.index.unique()
    
    # Convert obs index to a DataFrame for faster filtering
    obs_df = adata.obs.reset_index()
    
    # Iterate over unique keys with a progress bar
    for key in tqdm.tqdm(unique_keys, desc="Splitting AnnData by Keys"):
        # Efficient subsetting
        mask = obs_df['index'] == key
        indices = mask[mask].index
        
        # Subset AnnData without unnecessary copying
        adata_patch = adata[indices]
        
        # Reset obs index
        adata_patch.obs.index = adata_patch.obs.index.str.rsplit('-', n=1).str[-1]
        
        # Safely assign spatial information if available
        adata_patch.uns['spatial'] = aligned_dict.get(key, {}).get('spatial')
        adata_patch.uns['patch_coords'] = aligned_dict.get(key, {}).get('patch_coords')
        
        # Store the split AnnData object in the dictionary
        adata_patches[key] = adata_patch
    
    return adata_patches



# split the patches
adata_visium_pred_patches = split_adata_by_key(adata_visium_pred, aligned_visium_dictionary)
adata_xenium_pred_patches = split_adata_by_key(adata_xenium_pred, aligned_xenium_dictionary)


# plotRaster(adata_visium_pred.uns["spatial"], adata_visium_pred_patches, color_by='gene_expression', gene_name=g)


# New resolution that doesn't cause issues

# Rerasterize the patches
aligned_visium_dictionary_pred = rerasterize_patches(adata_visium_pred_patches, 275)
aligned_xenium_dictionary_pred = rerasterize_patches(adata_xenium_pred_patches, 275)


# save the patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_pred_rerasterized275.pkl', 'wb') as f:
    pickle.dump(aligned_visium_dictionary_pred, f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches_pred_rerasterized275.pkl', 'wb') as f:
    pickle.dump(aligned_xenium_dictionary_pred, f)