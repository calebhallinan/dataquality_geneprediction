
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


###########################################################################################################################


### Supp figure 1 ###


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data - Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["rMSE_range"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["rMSE_range"])
visium_mean = np.mean(visium_visimage_corr["rMSE_range"])
# plt.axvline(xenium_mean, color="C0", linestyle="--", label="Mean Xenium - Visium Image")
# plt.axvline(visium_mean, color="C1", linestyle="--", label="Mean Visium - Visium Image")
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
plt.savefig("fig1a_histogram_rmse.svg", dpi=1000, bbox_inches="tight")
# plt.savefig('fig2a.eps', format='eps')




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
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Normalized rMSE (mean)")
sns.despine()
# plt.legend()
plt.savefig("fig1b_rmse.svg", dpi=1000, bbox_inches="tight")



g = "ANKRD30A"
visium_visimage_corr[visium_visimage_corr['Gene'] == g][mse_type]
xenium_xenimage_corr[xenium_xenimage_corr['Gene'] == g][mse_type]


###########################################################################################################################


### Supp figure 2 ###



xenium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_test_per_patch_results_none.csv", index_col=0)
xenium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed0_test_per_patch_results_none.csv", index_col=0)
xenium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed1_test_per_patch_results_none.csv", index_col=0)
xenium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed10_test_per_patch_results_none.csv", index_col=0)
xenium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed100_test_per_patch_results_none.csv", index_col=0)



visium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_test_per_patch_results_none.csv", index_col=0)
visium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed0_test_per_patch_results_none.csv", index_col=0)
visium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed1_test_per_patch_results_none.csv", index_col=0)
visium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed10_test_per_patch_results_none.csv", index_col=0)
visium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed100_test_per_patch_results_none.csv", index_col=0)



gene = "GZMK" # ANKRD30A, GZMK, HDC, AHSP
# plot a gene scatterplot of the correlation values
plt.figure(figsize=(10, 5))
plt.scatter(visium_visimage_corr42[gene], xenium_xenimage_corr42[gene], alpha=1, c="black")
# plot line of 0,1
plt.plot([0, np.ceil(max(visium_visimage_corr42[gene].max(), xenium_xenimage_corr42[gene].max()))], [0, np.ceil(max(visium_visimage_corr42[gene].max(), xenium_xenimage_corr42[gene].max()))], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Gene: {}".format(gene))
sns.despine()
# plot correlation value on the plot
correlation_value = np.corrcoef(visium_visimage_corr42[gene], xenium_xenimage_corr42[gene])[0, 1]
plt.text(0.1, 0.9, f"Pearson r: {np.round(correlation_value, 3)}", 
         fontsize=10, color="black", ha="left", va="top",
            transform=plt.gca().transAxes)





# read in packages
import scanpy as sc
import scipy.sparse
import pickle
import numpy as np




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


# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)



# Get all patches with indices matching visium_visimage_corr42.index, prefixed with "patch_"
matching_patches = [patch for patch in aligned_visium_dictionary if patch in ["patch_" + str(idx) for idx in visium_visimage_corr42.index]]
matching_patches

matching_patch_objects = [aligned_visium_dictionary[patch] for patch in matching_patches]


# now grab all genes from each patch and make a dataframe of the gene expression values
patch_gene_expression_visium = pd.DataFrame()
for patch in matching_patch_objects:
    # get the gene expression values for each patch
    patch_gene_expression_visium = pd.concat([patch_gene_expression_visium, pd.DataFrame(patch.X.toarray())], axis=0)

patch_gene_expression_visium.index = visium_visimage_corr42.index
patch_gene_expression_visium.columns = visium_visimage_corr42.columns




gene = "ANKRD30A" # ANKRD30A, GZMK, HDC, AHSP

genes_to_plot = ["ANKRD30A", "GZMK", "HDC", "AHSP"]


for i, gene in enumerate(genes_to_plot):
    # plot a gene scatterplot of the correlation values
    plt.figure(figsize=(10, 5))
    x = patch_gene_expression_visium[gene]
    y = visium_visimage_corr42[gene]
    plt.scatter(x, y, alpha=1, c="black")
    # plot line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color="red", linestyle="--", lw=2, label="Best fit")
    plt.xlabel("Ground Truth: Visium data - Visium Image")
    plt.ylabel("Predicted: Visium data - Visium Image")
    plt.title("Gene: {}".format(gene))
    plt.xlim(0, np.ceil(max(y.max(), x.max())))
    plt.ylim(0, np.ceil(max(y.max(), x.max())))
    sns.despine()
    # plot correlation value on the plot
    correlation_value = np.corrcoef(y, x)[0, 1]
    plt.text(0.1, 0.9, f"Pearson r: {np.round(correlation_value, 3)}", 
            fontsize=10, color="black", ha="left", va="top",
                transform=plt.gca().transAxes)
    # save the plot
    plt.savefig(f"fig2_{gene}_visium_scatterplot.svg", dpi=1000, bbox_inches="tight")



# now do the same for xenium data
matching_patches = [patch for patch in aligned_xenium_dictionary if patch in ["patch_" + str(idx) for idx in xenium_xenimage_corr42.index]]
matching_patches

matching_patch_objects = [aligned_xenium_dictionary[patch] for patch in matching_patches]
# now grab all genes from each patch and make a dataframe of the gene expression values
patch_gene_expression_xenium = pd.DataFrame()
for patch in matching_patch_objects:
    # get the gene expression values for each patch
    patch_gene_expression_xenium = pd.concat([patch_gene_expression_xenium, pd.DataFrame(patch.X.toarray())], axis=0)
patch_gene_expression_xenium.index = xenium_xenimage_corr42.index
patch_gene_expression_xenium.columns = xenium_xenimage_corr42.columns


for i, gene in enumerate(genes_to_plot):
    # plot a gene scatterplot of the correlation values
    plt.figure(figsize=(10, 5))
    x = patch_gene_expression_xenium[gene]
    y = xenium_xenimage_corr42[gene]
    plt.scatter(x, y, alpha=1, c="black")
    # plot line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color="red", linestyle="--", lw=2, label="Best fit")
    plt.xlabel("Ground Truth: Xenium data - Xenium Image")
    plt.ylabel("Predicted: Xenium data - Xenium Image")
    plt.title("Gene: {}".format(gene))
    plt.xlim(0, np.ceil(max(y.max(), x.max())))
    plt.ylim(0, np.ceil(max(y.max(), x.max())))
    sns.despine()
    # plot correlation value on the plot
    correlation_value = np.corrcoef(y, x)[0, 1]
    plt.text(0.1, 0.9, f"Pearson r: {np.round(correlation_value, 3)}", 
            fontsize=10, color="black", ha="left", va="top",
                transform=plt.gca().transAxes)
    # save the plot
    plt.savefig(f"fig2_{gene}_xenium_scatterplot.svg", dpi=1000, bbox_inches="tight")



###########################################################################################################################


### Supp figure 3 ###


### read in data and make tables ###

import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns


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


# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=xenium_visimage_corr["rMSE_range"], c="black", linewidth = 0)
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Visium Image")
plt.title("Normalized rMSE (range)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig3a_scatterplot.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_xenimage_corr["rMSE_range"], y=xenium_xenimage_corr["rMSE_range"], alpha=1, c="black",linewidth = 0)
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Xenium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Normalized rMSE (range)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig3b_scatterplot.svg", dpi=1000, bbox_inches="tight")





#######################################################################################################################################


### Supp figure 4 ###


### read in data and make tables ###

import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

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




# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=visium_xenimage_corr["rMSE_range"], c="black", linewidth = 0)
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Visium data - Xenium Image")
plt.title("Normalized rMSE (range)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig4a_scatterplot.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_visimage_corr["rMSE_range"], y=xenium_xenimage_corr["rMSE_range"], alpha=1, c="black",linewidth = 0)
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Xenium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Normalized rMSE (range)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig4b_scatterplot.svg", dpi=1000, bbox_inches="tight")



#######################################################################################################################################


### Supp figure 6 ###


### read in data and make tables ###

import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

# read in data #


xenium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_xeniumdata_xeniumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_xeniumdata_xeniumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_xeniumdata_xeniumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_xeniumdata_xeniumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr = pd.concat([xenium_xenimage_corr42, xenium_xenimage_corr0, xenium_xenimage_corr1, xenium_xenimage_corr10, xenium_xenimage_corr100])
xenium_xenimage_corr_all = xenium_xenimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_xenimage_corr = xenium_xenimage_corr_all.groupby("Gene").mean().reset_index()


visium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_visiumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_visiumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_visiumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_visiumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/off_targets/breastcancer_visiumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr = pd.concat([visium_visimage_corr42, visium_visimage_corr0, visium_visimage_corr1, visium_visimage_corr10, visium_visimage_corr100])
visium_visimage_corr_all = visium_visimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_visimage_corr = visium_visimage_corr_all.groupby("Gene").mean().reset_index()



# sanity check
np.mean(xenium_xenimage_corr["Pearson"]), np.mean(visium_visimage_corr["Pearson"])



# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data - Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="C0", linestyle="--", label="Mean Xenium - Visium Image")
# plt.axvline(visium_mean, color="C1", linestyle="--", label="Mean Visium - Visium Image")
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
plt.savefig("fig6a_scatterplot.svg", dpi=1000, bbox_inches="tight")


# sort the correlation values by gene   
visium_visimage_corr = visium_visimage_corr.sort_values("Gene")
xenium_xenimage_corr = xenium_xenimage_corr.sort_values("Gene")

# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["Pearson"], y=xenium_xenimage_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig6b_scatterplot.svg", dpi=1000, bbox_inches="tight")




# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data - Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["rMSE_range"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["rMSE_range"])
visium_mean = np.mean(visium_visimage_corr["rMSE_range"])
# plt.axvline(xenium_mean, color="C0", linestyle="--", label="Mean Xenium - Visium Image")
# plt.axvline(visium_mean, color="C1", linestyle="--", label="Mean Visium - Visium Image")
plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="C0", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="C1", ha="center")

plt.xlabel("Normalized rMSE (range)")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
# plt.xlim(0, 1)
plt.legend()
plt.savefig("fig6c_histogram.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=xenium_xenimage_corr["rMSE_range"], c="black", linewidth = 0)
plt.plot([0, 0.3], [0, 0.3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Normalized rMSE (range)")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig6d_scatterplot_rmse.svg", dpi=1000, bbox_inches="tight")
