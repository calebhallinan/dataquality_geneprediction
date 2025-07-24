
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from PIL import Image
from adjustText import adjust_text




# read in data #

xenium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr = pd.concat([xenium_visimage_corr42, xenium_visimage_corr0, xenium_visimage_corr1, xenium_visimage_corr10, xenium_visimage_corr100])
xenium_visimage_corr_all = xenium_visimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_visimage_corr = xenium_visimage_corr_all.groupby("Gene").mean().reset_index()


xenium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr = pd.concat([xenium_xenimage_corr42, xenium_xenimage_corr0, xenium_xenimage_corr1, xenium_xenimage_corr10, xenium_xenimage_corr100])
xenium_xenimage_corr_all = xenium_xenimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_xenimage_corr = xenium_xenimage_corr_all.groupby("Gene").mean().reset_index()


visium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr = pd.concat([visium_visimage_corr42, visium_visimage_corr0, visium_visimage_corr1, visium_visimage_corr10, visium_visimage_corr100])
visium_visimage_corr_all = visium_visimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_visimage_corr = visium_visimage_corr_all.groupby("Gene").mean().reset_index()


visium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr = pd.concat([visium_xenimage_corr42, visium_xenimage_corr0, visium_xenimage_corr1, visium_xenimage_corr10, visium_xenimage_corr100])
visium_xenimage_corr_all = visium_xenimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_xenimage_corr = visium_xenimage_corr_all.groupby("Gene").mean().reset_index()


# sanity check
np.mean(xenium_visimage_corr["Pearson"]), np.mean(xenium_xenimage_corr["Pearson"]), np.mean(visium_visimage_corr["Pearson"]), np.mean(visium_xenimage_corr["Pearson"])


visium_visimage_corr[visium_visimage_corr['Gene'] == "GZMK"]
xenium_xenimage_corr[xenium_xenimage_corr['Gene'] == "GZMK"]


########################################################################################################################


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data w/ Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data w/ Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="C0", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="C1", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="C0", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="C1", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("fig2a_histogram.svg", dpi=1000, bbox_inches="tight")


###

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data w/ Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["rMSE_range"], color="#E69F01", label="Xenium data w/ Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["rMSE_range"])
visium_mean = np.mean(visium_visimage_corr["rMSE_range"])
# plt.axvline(xenium_mean, color="C0", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="C1", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="C0", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="C1", ha="center")

plt.xlabel("MSE")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
# plt.xlim(0, 1)
plt.legend()
# save the plot
# plt.savefig("fig2a.png", dpi=1000, bbox_inches="tight")
plt.savefig("fig2a_histogram_rmse.svg", dpi=1000, bbox_inches="tight")
# plt.savefig('fig2a.eps', format='eps')


########################################################################################################################


# sort the correlation values by gene   
xenium_visimage_corr = xenium_visimage_corr.sort_values("Gene")
visium_visimage_corr = visium_visimage_corr.sort_values("Gene")
xenium_xenimage_corr = xenium_xenimage_corr.sort_values("Gene")
visium_xenimage_corr = visium_xenimage_corr.sort_values("Gene")



# plot scatterplot of correlation values
plt.figure(figsize=(10, 5))
plt.scatter(visium_visimage_corr["Pearson"], xenium_xenimage_corr["Pearson"], alpha=1, c="black")
# plot line of 0,1
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()

# label the points with the gene names in list
genes_names = ["HDC", "GZMK","AHSP","ANKRD30A"]

# genes_names = [
#     "ADGRE5", "ADH1B", "AKR1C3", "APOBEC3A", "AQP1", "C1QA", "CD79B", "CEACAM6",
#     "FCGR3A", "FLNB", "KLRC1", "KRT6B", "PDGFRA", "S100A4", "TOMM7", "TPD52", "TUBA4A", "TUBB2B"
# ]


# Example data for demonstration
texts = []


# Annotate the selected genes on the scatter plot
for gene in genes_names:
    y = xenium_xenimage_corr.loc[xenium_xenimage_corr["Gene"] == gene, "Pearson"].values[0]
    x = visium_visimage_corr.loc[visium_visimage_corr["Gene"] == gene, "Pearson"].values[0]
    
    plt.annotate(
        gene,
        (x, y),  # Point coordinates
        xytext=(x + 5, y + 5),  # Offset for the label
        textcoords='offset points',  # Interpret `xytext` as offset in points
        # arrowprops=dict(
        #     arrowstyle="->",  # Arrow style
        #     color="blue",
        #     lw=1  # Line width of the arrow
        # ),
        fontsize=12,
        color="green"
    )

# save the plot
# plt.savefig("fig2b.png", dpi=1000, bbox_inches="tight")
plt.savefig("fig2b_scatterplot.svg", dpi=1000, bbox_inches="tight")


# sort the correlation values by gene   
xenium_visimage_corr = xenium_visimage_corr.sort_values("Gene")
visium_visimage_corr = visium_visimage_corr.sort_values("Gene")
xenium_xenimage_corr = xenium_xenimage_corr.sort_values("Gene")
visium_xenimage_corr = visium_xenimage_corr.sort_values("Gene")


mse_type = "rMSE_range"
genes_names = ["HDC", "GZMK", "AHSP", "ANKRD30A"]

# Prepare mask for highlighted genes
highlight_mask = visium_visimage_corr["Gene"].isin(genes_names)

plt.figure(figsize=(10, 5))
# Plot all points in black
plt.scatter(
    visium_visimage_corr[mse_type],
    xenium_xenimage_corr[mse_type],
    alpha=1,
    c="black",
    label="Other genes"
)
# Plot highlighted genes in a different color
plt.scatter(
    visium_visimage_corr.loc[highlight_mask, mse_type],
    xenium_xenimage_corr.loc[highlight_mask, mse_type],
    alpha=1,
    c="green",
    label="Highlighted genes"
)
# plot line of 0,1
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("Normalized rMSE (mean)")
sns.despine()
# plt.legend()
plt.savefig("fig2b_rmse.svg", dpi=1000, bbox_inches="tight")


########################################################################################################################




# read in data #

xenium_visimage_corr_rep2_42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed42_rep2_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_rep2_0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed0_rep2_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_rep2_1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed1_rep2_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_rep2_10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed10_rep2_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_rep2_100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_visiumimage_seed100_rep2_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_rep2 = pd.concat([xenium_visimage_corr_rep2_42, xenium_visimage_corr_rep2_0, xenium_visimage_corr_rep2_1, xenium_visimage_corr_rep2_10, xenium_visimage_corr_rep2_100])
xenium_visimage_corr_rep2_all = xenium_visimage_corr_rep2.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_visimage_corr_rep2 = xenium_visimage_corr_rep2_all.groupby("Gene").mean().reset_index()

xenium_xenimage_corr_rep2_42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_rep2_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_rep2_0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed0_rep2_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_rep2_1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed1_rep2_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_rep2_10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed10_rep2_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_rep2_100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed100_rep2_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_rep2 = pd.concat([xenium_xenimage_corr_rep2_42, xenium_xenimage_corr_rep2_0, xenium_xenimage_corr_rep2_1, xenium_xenimage_corr_rep2_10, xenium_xenimage_corr_rep2_100])
xenium_xenimage_corr_rep2_all = xenium_xenimage_corr_rep2.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_xenimage_corr_rep2 = xenium_xenimage_corr_rep2_all.groupby("Gene").mean().reset_index()


visium_visimage_corr_rep2_42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_df_none.csv", index_col=0)
visium_visimage_corr_rep2_0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed0_rep2_correlation_df_none.csv", index_col=0)
visium_visimage_corr_rep2_1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed1_rep2_correlation_df_none.csv", index_col=0)
visium_visimage_corr_rep2_10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed10_rep2_correlation_df_none.csv", index_col=0)
visium_visimage_corr_rep2_100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed100_rep2_correlation_df_none.csv", index_col=0)
visium_visimage_corr_rep2 = pd.concat([visium_visimage_corr_rep2_42, visium_visimage_corr_rep2_0, visium_visimage_corr_rep2_1, visium_visimage_corr_rep2_10, visium_visimage_corr_rep2_100])
visium_visimage_corr_rep2_all = visium_visimage_corr_rep2.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_visimage_corr_rep2 = visium_visimage_corr_rep2_all.groupby("Gene").mean().reset_index()


visium_xenimage_corr_rep2_42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed42_rep2_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_rep2_0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed0_rep2_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_rep2_1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed1_rep2_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_rep2_10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed10_rep2_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_rep2_100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed100_rep2_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_rep2 = pd.concat([visium_xenimage_corr_rep2_42, visium_xenimage_corr_rep2_0, visium_xenimage_corr_rep2_1, visium_xenimage_corr_rep2_10, visium_xenimage_corr_rep2_100])
visium_xenimage_corr_rep2_all = visium_xenimage_corr_rep2.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_xenimage_corr_rep2 = visium_xenimage_corr_rep2_all.groupby("Gene").mean().reset_index()


# sanity check
np.mean(xenium_visimage_corr_rep2["Pearson"]), np.mean(xenium_xenimage_corr_rep2["Pearson"]), np.mean(visium_visimage_corr_rep2["Pearson"]), np.mean(visium_xenimage_corr_rep2["Pearson"])





# sort the correlation values by gene   
xenium_visimage_corr_rep2 = xenium_visimage_corr_rep2.sort_values("Gene")
visium_visimage_corr_rep2 = visium_visimage_corr_rep2.sort_values("Gene")
xenium_xenimage_corr_rep2 = xenium_xenimage_corr_rep2.sort_values("Gene")
visium_xenimage_corr_rep2 = visium_xenimage_corr_rep2.sort_values("Gene")




# plot scatterplot of correlation values
plt.figure(figsize=(10, 5))
plt.scatter(np.sqrt(visium_visimage_corr_rep2["MSE"]), np.sqrt(xenium_xenimage_corr_rep2["MSE"]), alpha=1, c="black")
# plot line of 0,1
plt.plot([0, 3], [0, 3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("rMSE")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()







# plot scatterplot of correlation values
plt.figure(figsize=(10, 5))
scatter = plt.scatter(visium_visimage_corr_rep2["Pearson"], xenium_xenimage_corr_rep2["Pearson"], alpha=1, c='black')
# plot line of 0,1
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()


########################################################################################################################

import sys
sys.path.append('..')
from plotting_utils import *
import pickle
import scanpy as sc
import scipy.sparse


# read in data


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

# plot the patches

g = "LUM"

plotRaster(adata_visium.uns["spatial"], aligned_visium_dictionary, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium.uns["spatial"], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)
# plotRaster(adata_xenium.uns["spatial"], aligned_xenium_dictionary, color_by='gene_expression', gene_name=g)


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
        print(idx)
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

aligned_visium_dictionary_rerastered = rerasterize_patches(aligned_visium_dictionary, new_resolution_xenium)
aligned_xenium_dictionary_rerastered = rerasterize_patches(aligned_xenium_dictionary, new_resolution_xenium)


plotRaster(adata_visium.uns["spatial"], aligned_visium_dictionary_rerastered, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium.uns["spatial"], aligned_xenium_dictionary_rerastered, color_by='gene_expression', gene_name=g)


########################################################################################################################

# looking into individal patches

# for patch in aligned_visium_dictionary:
#     if aligned_visium_dictionary[patch].obsm["spatial"][0,0] == 1482 and aligned_visium_dictionary[patch].obsm["spatial"][0,1] == 1424:
#         print(i)
#         break

# for patch in aligned_xenium_dictionary:
#     if aligned_xenium_dictionary[patch].obsm["spatial"][0,0] == 4798 and aligned_xenium_dictionary[patch].obsm["spatial"][0,1] == 18075:
#         print(i)
#         break


# plt.imshow(aligned_visium_dictionary['patch_4990'].uns['spatial'])
# plt.imshow(aligned_xenium_dictionary['patch_4990'].uns['spatial'])


# plt.scatter(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1])
# plt.scatter(adata_visium.obsm["spatial"][100,0], adata_visium.obsm["spatial"][100,1], c="red")


# plt.scatter(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1])
# plt.scatter(adata_xenium.obsm["spatial"][100,0], adata_xenium.obsm["spatial"][100,1], c="red")


# def find_highest_y_lowest_x_with_index(x_values, y_values):
#     """
#     Finds the (x, y) pair with the highest y value and, in case of ties,
#     the lowest x value, along with its index in the original arrays.

#     Parameters:
#         x_values (list or array): List or array of x values.
#         y_values (list or array): List or array of y values.

#     Returns:
#         tuple: The (x, y) pair and its index as (x, y, index).
#     """
#     # Combine indices, x, and y into tuples
#     indexed_pairs = list(enumerate(zip(x_values, y_values)))
    
#     # Sort by y descending, then by x ascending
#     indexed_pairs_sorted = sorted(indexed_pairs, key=lambda pair: (-pair[1][1], pair[1][0]))
    
#     # Get the best pair
#     best_index, (best_x, best_y) = indexed_pairs_sorted[0]
    
#     return best_x, best_y, best_index



# # find the highest y and lowest x
# find_highest_y_lowest_x_with_index(adata_visium.obsm["spatial"][:,0], adata_visium.obsm["spatial"][:,1])

# find_highest_y_lowest_x_with_index(adata_xenium.obsm["spatial"][:,0], adata_xenium.obsm["spatial"][:,1])

# adata_visium[adata_visium.obsm["spatial"]]




# find the index of the patch with the highest y and lowest x


########################################################################################################################




# read in adata
adata_visium_pred = sc.read_h5ad("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visium_imagevisium_seed42_aligned_adata_pred_full_none.h5ad")
adata_xenium_pred = sc.read_h5ad("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xenium_imagexenium_seed42_aligned_adata_pred_full_none.h5ad")


# split the patches
# NOTE: this takes a LONG time! can read in the data instead
adata_visium_pred_patches = split_adata_patches(adata_visium_pred, aligned_visium_dictionary)
adata_xenium_pred_patches = split_adata_patches(adata_xenium_pred, aligned_xenium_dictionary)

# # # save the patches
# with open('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visium_imagevisium_seed42_aligned_adata_pred_full_none.pkl', 'wb') as f:
#     pickle.dump(adata_visium_pred_patches, f)

# with open('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xenium_imagexenium_seed42_aligned_adata_pred_full_none.pkl', 'wb') as f:
#     pickle.dump(adata_xenium_pred_patches, f)


# read in the rerastered patches
with open('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visium_imagevisium_seed42_aligned_adata_pred_full_none.pkl', 'rb') as f:
    adata_visium_pred_patches = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xenium_imagexenium_seed42_aligned_adata_pred_full_none.pkl', 'rb') as f:
    adata_xenium_pred_patches = pickle.load(f)

# Rerasterize the patches
aligned_visium_dictionary_pred = rerasterize_patches(adata_visium_pred_patches, 275)
aligned_xenium_dictionary_pred = rerasterize_patches(adata_xenium_pred_patches, 275)




# # read in the rerastered patches
# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches_rerastered.pkl', 'rb') as f:
#     aligned_visium_dictionary_pred = pickle.load(f)

# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches_rerastered.pkl', 'rb') as f:
#     aligned_xenium_dictionary_pred = pickle.load(f)


# plot the patches TEST to see if heatmap is correct
plotRaster(adata_visium_pred.uns["spatial"], aligned_visium_dictionary_pred, color_by='gene_expression', gene_name=g)
plotRaster(adata_xenium_pred.uns["spatial"], aligned_xenium_dictionary_pred, color_by='gene_expression', gene_name=g)

# plotRaster(adata_visium_pred.uns["spatial"], aligned_visium_dictionary_rerastered, color_by='gene_expression', gene_name=g)
# plotRaster(adata_visium_pred.uns["spatial"], aligned_visium_dictionary_pred, color_by='gene_expression', gene_name=g)




# genes_names = ["HDC", "REXO4","APOBEC3A","ANKRD30A"]
genes_names = ["HDC", "GZMK","AHSP","ANKRD30A"]


g = "ANKRD30A"

# plot the patches 
plotRasterSideBySide(adata_visium_pred.uns["spatial"], aligned_visium_dictionary_rerastered, adata_visium_pred.uns["spatial"], aligned_visium_dictionary_pred, color_by='gene_expression', gene_name=g, save_scheme="fig2c_" + g + "_visium")
plotRasterSideBySide(adata_xenium_pred.uns["spatial"], aligned_xenium_dictionary_rerastered, adata_xenium_pred.uns["spatial"], aligned_xenium_dictionary_pred, color_by='gene_expression', gene_name=g, save_scheme="fig2c_" + g + "_xenium")







# Function to plot patches on two images side by side with a shared color bar
def plotRasterSideBySide(image1, adata_patches1, image2, adata_patches2, color_by='gene_expression', gene_name=None, save_scheme="fig2c"):
    """
    Plots patches on two images side by side, colored by either gene expression or a column in adata_patches.obs.
    A single shared heatmap legend is used.

    Parameters:
    - image1: The first original image array.
    - adata_patches1: Dictionary of AnnData objects representing the patches for the first image.
    - image2: The second original image array.
    - adata_patches2: Dictionary of AnnData objects representing the patches for the second image.
    - color_by: How to color the patches ('gene_expression' or 'total_expression').
    - gene_name: The name of the gene to use if color_by is 'gene_expression'.
    """
    if color_by == 'gene_expression' and gene_name is None:
        raise ValueError("You must specify a gene_name when color_by='gene_expression'.")

    # Collect values across both adata_patches1 and adata_patches2 for normalization
    values = []
    for adata_patch in list(adata_patches1.values()) + list(adata_patches2.values()):
        if color_by == 'gene_expression':
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            values.append(expression)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            values.append(total_expression)
    
    # Determine color normalization range
    values = np.array(values)
    min_value, max_value = values.min(), values.max()

    # Set up subplots for side-by-side images
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Helper function to plot patches on each axis
    def plot_patches_on_image(ax, image, adata_patches, title=''):
        ax.imshow(image)
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
            
            # Draw rectangle for patch
            rect = mpatches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                      linewidth=1, edgecolor='none', facecolor=color, alpha=1)
            ax.add_patch(rect)
        # add title
        ax.set_title(title)
        ax.axis('off')

    # Plot patches on the first image
    plot_patches_on_image(axes[0], image1, adata_patches1)
    
    # Plot patches on the second image
    plot_patches_on_image(axes[1], image2, adata_patches2)
    
    # Create a single color bar for both images
    norm = plt.Normalize(min_value, max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.1, pad=0.02, shrink = .75)
    cbar.set_label(f'{gene_name} Expression' if color_by == 'gene_expression' else "Total Expression")

    # plt.savefig(save_scheme + ".png", dpi=1000, bbox_inches="tight")
    plt.savefig(save_scheme + ".svg", dpi=1000, bbox_inches="tight")
    # plt.show()


########################################################################################################################

