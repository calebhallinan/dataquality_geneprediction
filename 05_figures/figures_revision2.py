
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from PIL import Image
from adjustText import adjust_text


########################################################################################################################



#### HCC datasets ###


# read in data #

visiumhd_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_HCC_patchsize210_seed42_test_correlation_df.csv", index_col=0)
visiumhd_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_HCC_patchsize210_seed1_test_correlation_df.csv", index_col=0)
visiumhd_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_HCC_patchsize210_seed10_test_correlation_df.csv", index_col=0)
visiumhd_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_HCC_patchsize210_seed100_test_correlation_df.csv", index_col=0)
visiumhd_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_HCC_patchsize210_seed0_test_correlation_df.csv", index_col=0)
visiumhd_corr = pd.concat([visiumhd_corr42, visiumhd_corr0, visiumhd_corr1, visiumhd_corr10, visiumhd_corr100])
# visiumhd_corr = pd.concat([visiumhd_corr42])

visiumhd_corr_all = visiumhd_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visiumhd_corr = visiumhd_corr_all.groupby("Gene").mean().reset_index()


xenium_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_HCC_patchsize250_seed42_test_correlation_df.csv", index_col=0)
xenium_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_HCC_patchsize250_seed1_test_correlation_df.csv", index_col=0)
xenium_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_HCC_patchsize250_seed10_test_correlation_df.csv", index_col=0)
xenium_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_HCC_patchsize250_seed100_test_correlation_df.csv", index_col=0)
xenium_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_HCC_patchsize250_seed0_test_correlation_df.csv", index_col=0)
xenium_corr = pd.concat([xenium_corr42, xenium_corr0, xenium_corr1, xenium_corr10, xenium_corr100])
# xenium_corr = pd.concat([xenium_corr42])

xenium_corr_all = xenium_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_corr = xenium_corr_all.groupby("Gene").mean().reset_index()


cosmx_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_HCC_patchsize250_seed42_test_correlation_df.csv", index_col=0)
cosmx_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_HCC_patchsize250_seed1_test_correlation_df.csv", index_col=0)
cosmx_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_HCC_patchsize250_seed10_test_correlation_df.csv", index_col=0)
cosmx_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_HCC_patchsize250_seed100_test_correlation_df.csv", index_col=0)
cosmx_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_HCC_patchsize250_seed0_test_correlation_df.csv", index_col=0)
cosmx_corr = pd.concat([cosmx_corr42, cosmx_corr0, cosmx_corr1, cosmx_corr10, cosmx_corr100])
# cosmx_corr = pd.concat([cosmx_corr42])

cosmx_corr_all = cosmx_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
cosmx_corr = cosmx_corr_all.groupby("Gene").mean().reset_index()


stereoseq_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_HCC_patchsize110_seed42_test_correlation_df.csv", index_col=0)
stereoseq_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_HCC_patchsize110_seed1_test_correlation_df.csv", index_col=0)
stereoseq_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_HCC_patchsize110_seed10_test_correlation_df.csv", index_col=0)
stereoseq_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_HCC_patchsize110_seed100_test_correlation_df.csv", index_col=0)
stereoseq_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_HCC_patchsize110_seed0_test_correlation_df.csv", index_col=0)
stereoseq_corr = pd.concat([stereoseq_corr42, stereoseq_corr0, stereoseq_corr1, stereoseq_corr10, stereoseq_corr100])
# stereoseq_corr = pd.concat([stereoseq_corr42])

stereoseq_corr_all = stereoseq_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
stereoseq_corr = stereoseq_corr_all.groupby("Gene").mean().reset_index()



# xenium_corr[xenium_corr['Gene'] == "MAP4"]
# visiumhd_corr[visiumhd_corr['Gene'] == "MAP4"]
# cosmx_corr[cosmx_corr['Gene'] == "MAP4"]
# stereoseq_corr[stereoseq_corr['Gene'] == "MAP4"]


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["Pearson"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["Pearson"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["Pearson"], color="C6", label="CosMx", kde=True)
sns.histplot(stereoseq_corr["Pearson"], color="C7", label="Stereo-seq", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["Pearson"])
visium_mean = np.mean(visiumhd_corr["Pearson"])
cosmx_mean = np.mean(cosmx_corr["Pearson"])
stereoseq_mean = np.mean(stereoseq_corr["Pearson"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")
plt.axvline(stereoseq_mean, color="C7", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")
plt.text(stereoseq_mean, plt.ylim()[1]*1, f"{np.round(stereoseq_mean, 3)}", 
         fontsize=10, color="C7", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig_HCC_histogram1.svg", dpi=1000, bbox_inches="tight")


# RMSE


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["rMSE_range"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["rMSE_range"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["rMSE_range"], color="C6", label="CosMx", kde=True)
sns.histplot(stereoseq_corr["rMSE_range"], color="C7", label="Stereo-seq", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["rMSE_range"])
visium_mean = np.mean(visiumhd_corr["rMSE_range"])
cosmx_mean = np.mean(cosmx_corr["rMSE_range"])
stereoseq_mean = np.mean(stereoseq_corr["rMSE_range"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")
plt.axvline(stereoseq_mean, color="C7", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")
plt.text(stereoseq_mean, plt.ylim()[1]*1, f"{np.round(stereoseq_mean, 3)}", 
         fontsize=10, color="C7", ha="center")

plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig_HCC_histogram2.svg", dpi=1000, bbox_inches="tight")




# align datasets by gene name
visiumhd_corr = visiumhd_corr.sort_values("Gene").reset_index(drop=True)
xenium_corr = xenium_corr.sort_values("Gene").reset_index(drop=True)
cosmx_corr = cosmx_corr.sort_values("Gene").reset_index(drop=True)
stereoseq_corr = stereoseq_corr.sort_values("Gene").reset_index(drop=True)

common_genes = sorted(
    set(visiumhd_corr["Gene"])
    .intersection(xenium_corr["Gene"])
    .intersection(cosmx_corr["Gene"])
    .intersection(stereoseq_corr["Gene"])
)

len(common_genes)

visiumhd_corr = visiumhd_corr[visiumhd_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
xenium_corr = xenium_corr[xenium_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
cosmx_corr = cosmx_corr[cosmx_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
stereoseq_corr = stereoseq_corr[stereoseq_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)




# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=xenium_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("Xenium")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot1.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=cosmx_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("CosMx")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot2.svg", dpi=1000, bbox_inches="tight")


plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["Pearson"], y=cosmx_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Xenium")
plt.ylabel("CosMx")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



#### more scatterplots ###


plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Xenium")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



plt.figure(figsize=(10, 5))
sns.scatterplot(x=cosmx_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("CosMx")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")







########################################################################################################################



#### OV datasets ###


# read in data #

visiumhd_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_OV_patchsize210_seed42_test_correlation_df.csv", index_col=0)
visiumhd_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_OV_patchsize210_seed1_test_correlation_df.csv", index_col=0)
visiumhd_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_OV_patchsize210_seed10_test_correlation_df.csv", index_col=0)
visiumhd_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_OV_patchsize210_seed100_test_correlation_df.csv", index_col=0)
visiumhd_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/visiumhd_OV_patchsize210_seed0_test_correlation_df.csv", index_col=0)
visiumhd_corr = pd.concat([visiumhd_corr42, visiumhd_corr0, visiumhd_corr1, visiumhd_corr10, visiumhd_corr100])
# visiumhd_corr = pd.concat([visiumhd_corr42])

visiumhd_corr_all = visiumhd_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visiumhd_corr = visiumhd_corr_all.groupby("Gene").mean().reset_index()


xenium_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_OV_patchsize250_seed42_test_correlation_df.csv", index_col=0)
xenium_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_OV_patchsize250_seed1_test_correlation_df.csv", index_col=0)
xenium_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_OV_patchsize250_seed10_test_correlation_df.csv", index_col=0)
xenium_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_OV_patchsize250_seed100_test_correlation_df.csv", index_col=0)
xenium_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/xenium_OV_patchsize250_seed0_test_correlation_df.csv", index_col=0)
xenium_corr = pd.concat([xenium_corr42, xenium_corr0, xenium_corr1, xenium_corr10, xenium_corr100])
# xenium_corr = pd.concat([xenium_corr42])

xenium_corr_all = xenium_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_corr = xenium_corr_all.groupby("Gene").mean().reset_index()


cosmx_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_OV_patchsize250_seed42_test_correlation_df.csv", index_col=0)
cosmx_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_OV_patchsize250_seed1_test_correlation_df.csv", index_col=0)
cosmx_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_OV_patchsize250_seed10_test_correlation_df.csv", index_col=0)
cosmx_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_OV_patchsize250_seed100_test_correlation_df.csv", index_col=0)
cosmx_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/cosmx_OV_patchsize250_seed0_test_correlation_df.csv", index_col=0)
cosmx_corr = pd.concat([cosmx_corr42, cosmx_corr0, cosmx_corr1, cosmx_corr10, cosmx_corr100])
# cosmx_corr = pd.concat([cosmx_corr42])

cosmx_corr_all = cosmx_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
cosmx_corr = cosmx_corr_all.groupby("Gene").mean().reset_index()


stereoseq_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_OV_patchsize110_seed42_test_correlation_df.csv", index_col=0)
stereoseq_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_OV_patchsize110_seed1_test_correlation_df.csv", index_col=0)
stereoseq_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_OV_patchsize110_seed10_test_correlation_df.csv", index_col=0)
stereoseq_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_OV_patchsize110_seed100_test_correlation_df.csv", index_col=0)
stereoseq_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/08_revision2/results/stereoseq_OV_patchsize110_seed0_test_correlation_df.csv", index_col=0)
stereoseq_corr = pd.concat([stereoseq_corr42, stereoseq_corr0, stereoseq_corr1, stereoseq_corr10, stereoseq_corr100])
# stereoseq_corr = pd.concat([stereoseq_corr42])

stereoseq_corr_all = stereoseq_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
stereoseq_corr = stereoseq_corr_all.groupby("Gene").mean().reset_index()



# xenium_corr[xenium_corr['Gene'] == "MAP4"]
# visiumhd_corr[visiumhd_corr['Gene'] == "MAP4"]
# cosmx_corr[cosmx_corr['Gene'] == "MAP4"]
# stereoseq_corr[stereoseq_corr['Gene'] == "MAP4"]


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["Pearson"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["Pearson"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["Pearson"], color="C6", label="CosMx", kde=True)
sns.histplot(stereoseq_corr["Pearson"], color="C7", label="Stereo-seq", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["Pearson"])
visium_mean = np.mean(visiumhd_corr["Pearson"])
cosmx_mean = np.mean(cosmx_corr["Pearson"])
stereoseq_mean = np.mean(stereoseq_corr["Pearson"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")
plt.axvline(stereoseq_mean, color="C7", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")
plt.text(stereoseq_mean, plt.ylim()[1]*1, f"{np.round(stereoseq_mean, 3)}", 
         fontsize=10, color="C7", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig_OV_histogram1.svg", dpi=1000, bbox_inches="tight")


# RMSE


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["rMSE_range"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["rMSE_range"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["rMSE_range"], color="C6", label="CosMx", kde=True)
sns.histplot(stereoseq_corr["rMSE_range"], color="C7", label="Stereo-seq", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["rMSE_range"])
visium_mean = np.mean(visiumhd_corr["rMSE_range"])
cosmx_mean = np.mean(cosmx_corr["rMSE_range"])
stereoseq_mean = np.mean(stereoseq_corr["rMSE_range"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")
plt.axvline(stereoseq_mean, color="C7", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")
plt.text(stereoseq_mean, plt.ylim()[1]*1, f"{np.round(stereoseq_mean, 3)}", 
         fontsize=10, color="C7", ha="center")

plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig_OV_histogram2.svg", dpi=1000, bbox_inches="tight")




# align datasets by gene name
visiumhd_corr = visiumhd_corr.sort_values("Gene").reset_index(drop=True)
xenium_corr = xenium_corr.sort_values("Gene").reset_index(drop=True)
cosmx_corr = cosmx_corr.sort_values("Gene").reset_index(drop=True)
stereoseq_corr = stereoseq_corr.sort_values("Gene").reset_index(drop=True)

common_genes = sorted(
    set(visiumhd_corr["Gene"])
    .intersection(xenium_corr["Gene"])
    .intersection(cosmx_corr["Gene"])
    .intersection(stereoseq_corr["Gene"])
)

len(common_genes)

visiumhd_corr = visiumhd_corr[visiumhd_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
xenium_corr = xenium_corr[xenium_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
cosmx_corr = cosmx_corr[cosmx_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)
stereoseq_corr = stereoseq_corr[stereoseq_corr["Gene"].isin(common_genes)].sort_values("Gene").reset_index(drop=True)




# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=xenium_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("Xenium")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot1.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=cosmx_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("CosMx")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot2.svg", dpi=1000, bbox_inches="tight")


plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["Pearson"], y=cosmx_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Xenium")
plt.ylabel("CosMx")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



#### more scatterplots ###


plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Xenium")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



plt.figure(figsize=(10, 5))
sns.scatterplot(x=cosmx_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("CosMx")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



plt.figure(figsize=(10, 5))
sns.scatterplot(x=visiumhd_corr["Pearson"], y=stereoseq_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("VisiumHD")
plt.ylabel("Stereo-seq")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/08_revision2/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")

