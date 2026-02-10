
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from PIL import Image
from adjustText import adjust_text


########################################################################################################################

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


### higher visium resolution ###


# read in data #

xenium_visimage_corr42_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_xeniumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr0_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_xeniumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr1_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_xeniumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr10_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_xeniumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr100_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_xeniumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr_hires = pd.concat([xenium_visimage_corr42_hires, xenium_visimage_corr0_hires, xenium_visimage_corr1_hires, xenium_visimage_corr10_hires, xenium_visimage_corr100_hires])
xenium_visimage_corr_all_hires = xenium_visimage_corr_hires.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_visimage_corr_hires = xenium_visimage_corr_all_hires.groupby("Gene").mean().reset_index()


xenium_xenimage_corr42_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr0_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr1_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr10_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr100_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_xenimage_corr_hires = pd.concat([xenium_xenimage_corr42_hires, xenium_xenimage_corr0_hires, xenium_xenimage_corr1_hires, xenium_xenimage_corr10_hires, xenium_xenimage_corr100_hires])
xenium_xenimage_corr_all_hires = xenium_xenimage_corr_hires.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_xenimage_corr_hires = xenium_xenimage_corr_all_hires.groupby("Gene").mean().reset_index()


visium_visimage_corr42_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_visiumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr0_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_visiumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr1_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_visiumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr10_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_visiumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr100_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/tiff/breastcancer_visiumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr_hires = pd.concat([visium_visimage_corr42_hires, visium_visimage_corr0_hires, visium_visimage_corr1_hires, visium_visimage_corr10_hires, visium_visimage_corr100_hires])
visium_visimage_corr_all_hires = visium_visimage_corr_hires.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_visimage_corr_hires = visium_visimage_corr_all_hires.groupby("Gene").mean().reset_index()


visium_xenimage_corr42_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr0_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr1_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr10_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr100_hires = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_xeniumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_xenimage_corr_hires = pd.concat([visium_xenimage_corr42_hires, visium_xenimage_corr0_hires, visium_xenimage_corr1_hires, visium_xenimage_corr10_hires, visium_xenimage_corr100_hires])
visium_xenimage_corr_all_hires = visium_xenimage_corr_hires.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_xenimage_corr_hires = visium_xenimage_corr_all_hires.groupby("Gene").mean().reset_index()


# sanity check
np.mean(xenium_visimage_corr_hires["Pearson"]), np.mean(xenium_xenimage_corr_hires["Pearson"]), np.mean(visium_visimage_corr_hires["Pearson"]), np.mean(visium_xenimage_corr_hires["Pearson"])


# visium_visimage_corr[visium_visimage_corr['Gene'] == "GZMK"]
# xenium_xenimage_corr[xenium_xenimage_corr['Gene'] == "GZMK"]



# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data - High Resolution Visium Image", kde=True)
sns.histplot(visium_visimage_corr_hires["Pearson"], color="#005888", label="Visium data - WSI Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
visium_mean_hires = np.mean(visium_visimage_corr_hires["Pearson"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(visium_mean_hires, color="#005888", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(visium_mean_hires, plt.ylim()[1]*1, f"{np.round(visium_mean_hires, 3)}", 
         fontsize=10, color="#005888", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")


plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_histogram.svg", dpi=1000, bbox_inches="tight")


# visium vs xenium on visium image #

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data - High Resolution Visium Image", kde=True)
sns.histplot(visium_visimage_corr_hires["Pearson"], color="#005888", label="Visium data - WSI Visium Image", kde=True)
sns.histplot(xenium_visimage_corr["Pearson"], color="C3", label="Xenium data - High Resolution Visium Image", kde=True)
sns.histplot(xenium_visimage_corr_hires["Pearson"], color="#5D0002", label="Xenium data - WSI Visium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_visimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
xenium_mean_hires = np.mean(xenium_visimage_corr_hires["Pearson"])
visium_mean_hires = np.mean(visium_visimage_corr_hires["Pearson"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="C3", linestyle="--")
plt.axvline(visium_mean_hires, color="#005888", linestyle="--")
plt.axvline(xenium_mean_hires, color="#5D0002", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="C3", ha="center")
plt.text(visium_mean_hires, plt.ylim()[1]*1, f"{np.round(visium_mean_hires, 3)}", 
         fontsize=10, color="#005888", ha="center")
plt.text(xenium_mean_hires, plt.ylim()[1]*1, f"{np.round(xenium_mean_hires, 3)}", 
         fontsize=10, color="#5D0002", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_histogram2.svg", dpi=1000, bbox_inches="tight")



### RMSE ###





# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data - High Resolution Visium Image", kde=True)
sns.histplot(visium_visimage_corr_hires["rMSE_range"], color="#005888", label="Visium data - WSI Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["rMSE_range"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["rMSE_range"])
visium_mean = np.mean(visium_visimage_corr["rMSE_range"])
visium_mean_hires = np.mean(visium_visimage_corr_hires["rMSE_range"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(visium_mean_hires, color="#005888", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(visium_mean_hires, plt.ylim()[1]*1, f"{np.round(visium_mean_hires, 3)}", 
         fontsize=10, color="#005888", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")


plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_histogram3.svg", dpi=1000, bbox_inches="tight")


# visium vs xenium on visium image #

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data - High Resolution Visium Image", kde=True)
sns.histplot(visium_visimage_corr_hires["rMSE_range"], color="#005888", label="Visium data - WSI Visium Image", kde=True)
sns.histplot(xenium_visimage_corr["rMSE_range"], color="C3", label="Xenium data - High Resolution Visium Image", kde=True)
sns.histplot(xenium_visimage_corr_hires["rMSE_range"], color="#5D0002", label="Xenium data - WSI Visium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_visimage_corr["rMSE_range"])
visium_mean = np.mean(visium_visimage_corr["rMSE_range"])
xenium_mean_hires = np.mean(xenium_visimage_corr_hires["rMSE_range"])
visium_mean_hires = np.mean(visium_visimage_corr_hires["rMSE_range"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="C3", linestyle="--")
plt.axvline(visium_mean_hires, color="#005888", linestyle="--")
plt.axvline(xenium_mean_hires, color="#5D0002", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="C3", ha="center")
plt.text(visium_mean_hires, plt.ylim()[1]*1, f"{np.round(visium_mean_hires, 3)}", 
         fontsize=10, color="#005888", ha="center")
plt.text(xenium_mean_hires, plt.ylim()[1]*1, f"{np.round(xenium_mean_hires, 3)}", 
         fontsize=10, color="#5D0002", ha="center")

plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
# plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_histogram4.svg", dpi=1000, bbox_inches="tight")





# # scatterplot of xenium vs visium on visium image #

# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=visium_visimage_corr["Pearson"], y=xenium_visimage_corr["Pearson"], c="black", linewidth = 0)
# plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data - Visium Image")
# plt.ylabel("Xenium data - Visium Image")
# plt.title("Pearson Correlation Values")
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
# sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_scatterplot1.svg", dpi=1000, bbox_inches="tight")


# # scatterplot of xenium vs visium on xenium image #

# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=visium_visimage_corr["Pearson"], y=xenium_xenimage_corr["Pearson"], alpha=1, c="black",linewidth = 0)
# plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data - Visium Image")
# plt.ylabel("Xenium data - Xenium Image")
# plt.title("Pearson Correlation Values")
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
# sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_scatterplot2.svg", dpi=1000, bbox_inches="tight")




# # scatterplot of xenium vs visium on visium image # RMSE ##############

# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=xenium_visimage_corr["rMSE_range"], c="black", linewidth = 0)
# plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data - Visium Image")
# plt.ylabel("Xenium data - Visium Image")
# plt.title("rMSE_range")
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
# sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_scatterplot3.svg", dpi=1000, bbox_inches="tight")


# # scatterplot of xenium vs visium on xenium image #

# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=xenium_xenimage_corr["rMSE_range"], alpha=1, c="black",linewidth = 0)
# plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data - Visium Image")
# plt.ylabel("Xenium data - Xenium Image")
# plt.title("rMSE_range")
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
# sns.despine()
# plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig8_scatterplot4.svg", dpi=1000, bbox_inches="tight")



########################################################################################################################


### UNI results ###




# read in data #

xenium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
xenium_visimage_corr = pd.concat([xenium_visimage_corr42, xenium_visimage_corr0, xenium_visimage_corr1, xenium_visimage_corr10, xenium_visimage_corr100])
xenium_visimage_corr_all = xenium_visimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_visimage_corr = xenium_visimage_corr_all.groupby("Gene").mean().reset_index()


xenium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_df_UNI.csv", index_col=0)
xenium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_xeniumimage_seed0_test_correlation_df_UNI.csv", index_col=0)
xenium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_xeniumimage_seed1_test_correlation_df_UNI.csv", index_col=0)
xenium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_xeniumimage_seed10_test_correlation_df_UNI.csv", index_col=0)
xenium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_xeniumdata_xeniumimage_seed100_test_correlation_df_UNI.csv", index_col=0)
xenium_xenimage_corr = pd.concat([xenium_xenimage_corr42, xenium_xenimage_corr0, xenium_xenimage_corr1, xenium_xenimage_corr10, xenium_xenimage_corr100])
xenium_xenimage_corr_all = xenium_xenimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_xenimage_corr = xenium_xenimage_corr_all.groupby("Gene").mean().reset_index()


visium_visimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_visiumimage_seed42_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_visiumimage_seed0_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_visiumimage_seed1_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_visiumimage_seed10_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_visiumimage_seed100_test_correlation_df_none.csv", index_col=0)
visium_visimage_corr = pd.concat([visium_visimage_corr42, visium_visimage_corr0, visium_visimage_corr1, visium_visimage_corr10, visium_visimage_corr100])
visium_visimage_corr_all = visium_visimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_visimage_corr = visium_visimage_corr_all.groupby("Gene").mean().reset_index()


visium_xenimage_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_xeniumimage_seed42_test_correlation_df_UNI.csv", index_col=0)
visium_xenimage_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_xeniumimage_seed0_test_correlation_df_UNI.csv", index_col=0)
visium_xenimage_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_xeniumimage_seed1_test_correlation_df_UNI.csv", index_col=0)
visium_xenimage_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_xeniumimage_seed10_test_correlation_df_UNI.csv", index_col=0)
visium_xenimage_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/UNI/breastcancer_visiumdata_xeniumimage_seed100_test_correlation_df_UNI.csv", index_col=0)
visium_xenimage_corr = pd.concat([visium_xenimage_corr42, visium_xenimage_corr0, visium_xenimage_corr1, visium_xenimage_corr10, visium_xenimage_corr100])
visium_xenimage_corr_all = visium_xenimage_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visium_xenimage_corr = visium_xenimage_corr_all.groupby("Gene").mean().reset_index()


# sanity check
np.mean(xenium_visimage_corr["Pearson"]), np.mean(xenium_xenimage_corr["Pearson"]), np.mean(visium_visimage_corr["Pearson"]), np.mean(visium_xenimage_corr["Pearson"])


visium_visimage_corr[visium_visimage_corr['Gene'] == "GZMK"]
xenium_xenimage_corr[xenium_xenimage_corr['Gene'] == "GZMK"]


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data - Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)
# sns.histplot(xenium_visimage_corr["Pearson"], color="C3", label="Xenium data - Visium Image", kde=True)
# sns.histplot(visium_xenimage_corr["Pearson"], color="C4", label="Visium data - Xenium Image", kde=True)

# plot average correlation
xenium_xen_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_vis_mean = np.mean(visium_visimage_corr["Pearson"])
# xenium_vis_mean = np.mean(xenium_visimage_corr["Pearson"])
# visium_xen_mean = np.mean(visium_xenimage_corr["Pearson"])

plt.axvline(visium_vis_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_xen_mean, color="#E69F01", linestyle="--")
# plt.axvline(xenium_vis_mean, color="C3", linestyle="--")
# plt.axvline(visium_xen_mean, color="C4", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_vis_mean, plt.ylim()[1]*0.9, f"{np.round(visium_vis_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_xen_mean, plt.ylim()[1]*0.9, f"{np.round(xenium_xen_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
# plt.text(xenium_vis_mean, plt.ylim()[1]*0.9, f"{np.round(xenium_vis_mean, 3)}", 
#          fontsize=10, color="C3", ha="center")
# plt.text(visium_xen_mean, plt.ylim()[1]*0.9, f"{np.round(visium_xen_mean, 3)}", 
#          fontsize=10, color="C4", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig9_histogram1.svg", dpi=1000, bbox_inches="tight")


### RMSE ###


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["rMSE_range"], color="#55B4E9", label="Visium data - Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["rMSE_range"], color="#E69F01", label="Xenium data - Xenium Image", kde=True)
# sns.histplot(xenium_visimage_corr["rMSE_range"], color="C3", label="Xenium data - Visium Image", kde=True)
# sns.histplot(visium_xenimage_corr["rMSE_range"], color="C4", label="Visium data - Xenium Image", kde=True)

# plot average correlation
xenium_xen_mean = np.mean(xenium_xenimage_corr["rMSE_range"])
visium_vis_mean = np.mean(visium_visimage_corr["rMSE_range"])
# xenium_vis_mean = np.mean(xenium_visimage_corr["rMSE_range"])
# visium_xen_mean = np.mean(visium_xenimage_corr["rMSE_range"])

plt.axvline(visium_vis_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_xen_mean, color="#E69F01", linestyle="--")
# plt.axvline(xenium_vis_mean, color="C3", linestyle="--")
# plt.axvline(visium_xen_mean, color="C4", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_vis_mean, plt.ylim()[1]*0.9, f"{np.round(visium_vis_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_xen_mean, plt.ylim()[1]*0.9, f"{np.round(xenium_xen_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
# plt.text(xenium_vis_mean, plt.ylim()[1]*0.9, f"{np.round(xenium_vis_mean, 3)}", 
#          fontsize=10, color="C3", ha="center")
# plt.text(visium_xen_mean, plt.ylim()[1]*0.9, f"{np.round(visium_xen_mean, 3)}", 
#          fontsize=10, color="C4", ha="center")

plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig9_histogram2.svg", dpi=1000, bbox_inches="tight")


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
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig9_scatterplot1.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["rMSE_range"], y=xenium_xenimage_corr["rMSE_range"], alpha=1, c="black",linewidth = 0)
plt.plot([0, .3], [0, .3], color="gray", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("rMSE_range")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig9_scatterplot2.svg", dpi=1000, bbox_inches="tight")


########################################################################################################################


#### both datasets ###

# seed0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/bothdatasets/breastcancer_bothdata_bothimage_seed0_test_correlation_summary.txt", sep="\t", header=None)
# seed1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/bothdatasets/breastcancer_bothdata_bothimage_seed1_test_correlation_summary.txt", sep="\t", header=None)
# seed10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/bothdatasets/breastcancer_bothdata_bothimage_seed10_test_correlation_summary.txt", sep="\t", header=None)
# seed100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/bothdatasets/breastcancer_bothdata_bothimage_seed100_test_correlation_summary.txt", sep="\t", header=None)
# seed42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/results/bothdatasets/breastcancer_bothdata_bothimage_seed42_test_correlation_summary.txt", sep="\t", header=None)

# # Extract and convert the Pearson correlation values to float
# correlations = [
#     float(seed0.iloc[1, 0].split(":")[1].strip()),
#     float(seed1.iloc[1, 0].split(":")[1].strip()),
#     float(seed10.iloc[1, 0].split(":")[1].strip()),
#     float(seed100.iloc[1, 0].split(":")[1].strip()),
#     float(seed42.iloc[1, 0].split(":")[1].strip())
# ]

# # Calculate the average
# average_correlation = np.mean(correlations)
# print(f"Average Pearson Correlation: {average_correlation}")



########################################################################################################################


#### COAD datasets ###


# read in data #

visiumhd_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed42_test_correlation_df.csv", index_col=0)
visiumhd_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed1_test_correlation_df.csv", index_col=0)
visiumhd_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed10_test_correlation_df.csv", index_col=0)
visiumhd_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed100_test_correlation_df.csv", index_col=0)
visiumhd_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed0_test_correlation_df.csv", index_col=0)
visiumhd_corr = pd.concat([visiumhd_corr42, visiumhd_corr0, visiumhd_corr1, visiumhd_corr10, visiumhd_corr100])
visiumhd_corr_all = visiumhd_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visiumhd_corr = visiumhd_corr_all.groupby("Gene").mean().reset_index()


xenium_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed42_test_correlation_df.csv", index_col=0)
xenium_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed1_test_correlation_df.csv", index_col=0)
xenium_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed10_test_correlation_df.csv", index_col=0)
xenium_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed100_test_correlation_df.csv", index_col=0)
xenium_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed0_test_correlation_df.csv", index_col=0)
xenium_corr = pd.concat([xenium_corr42, xenium_corr0, xenium_corr1, xenium_corr10, xenium_corr100])
xenium_corr_all = xenium_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_corr = xenium_corr_all.groupby("Gene").mean().reset_index()


cosmx_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed42_test_correlation_df.csv", index_col=0)
cosmx_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed1_test_correlation_df.csv", index_col=0)
cosmx_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed10_test_correlation_df.csv", index_col=0)
cosmx_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed100_test_correlation_df.csv", index_col=0)
cosmx_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed0_test_correlation_df.csv", index_col=0)
cosmx_corr = pd.concat([cosmx_corr42, cosmx_corr0, cosmx_corr1, cosmx_corr10, cosmx_corr100])
cosmx_corr_all = cosmx_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
cosmx_corr = cosmx_corr_all.groupby("Gene").mean().reset_index()



xenium_corr[xenium_corr['Gene'] == "MAP4"]
visiumhd_corr[visiumhd_corr['Gene'] == "MAP4"]
cosmx_corr[cosmx_corr['Gene'] == "MAP4"]


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["Pearson"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["Pearson"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["Pearson"], color="C6", label="CosMx", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["Pearson"])
visium_mean = np.mean(visiumhd_corr["Pearson"])
cosmx_mean = np.mean(cosmx_corr["Pearson"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
sns.despine()
plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig10_histogram1.svg", dpi=1000, bbox_inches="tight")


# RMSE


# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visiumhd_corr["rMSE_range"], color="#55B4E9", label="VisiumHD", kde=True)
sns.histplot(xenium_corr["rMSE_range"], color="#E69F01", label="Xenium", kde=True)
sns.histplot(cosmx_corr["rMSE_range"], color="C6", label="CosMx", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_corr["rMSE_range"])
visium_mean = np.mean(visiumhd_corr["rMSE_range"])
cosmx_mean = np.mean(cosmx_corr["rMSE_range"])

plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")
plt.axvline(cosmx_mean, color="C6", linestyle="--")

# plot the average correlation of the datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean, 3)}", 
         fontsize=10, color="#E69F01", ha="center")
plt.text(cosmx_mean, plt.ylim()[1]*1, f"{np.round(cosmx_mean, 3)}", 
         fontsize=10, color="C6", ha="center")

plt.xlabel("rMSE_range")
plt.ylabel("Frequency")
sns.despine()
# plt.xlim(0, 1)
plt.legend()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig10_histogram2.svg", dpi=1000, bbox_inches="tight")




# align dataseets #


visiumhd_corr
xenium_corr
cosmx_corr


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
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig10_scatterplot1.svg", dpi=1000, bbox_inches="tight")


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
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig10_scatterplot2.svg", dpi=1000, bbox_inches="tight")


plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["Pearson"], y=cosmx_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
plt.xlabel("Xenium")
plt.ylabel("CosMx")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig10_scatterplot3.svg", dpi=1000, bbox_inches="tight")



g = "PPP1R1B"
cosmx_corr[cosmx_corr['Gene'] == g]
xenium_corr[xenium_corr['Gene'] == g]
visiumhd_corr[visiumhd_corr['Gene'] == g]











########################################################################################################################


#### marker gene plot ###


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


# import scanpy as sc
# import scipy.sparse


# # read in adatas
# adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
# adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data.h5ad')

# # make .X a csr matrix
# adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
# adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# # log the expression
# # log transform the data
# sc.pp.log1p(adata_xenium)
# sc.pp.log1p(adata_visium)

# # Calculate average gene expression for each gene
# adata_xenium.var["avg_expression"] = adata_xenium.X.mean(axis=0).A1
# adata_visium.var["avg_expression"] = adata_visium.X.mean(axis=0).A1



genes_names = ["FASN", "FOXA1", "CEACAM6", "GATA3", "MZB1", "AGR3", "SERPINA3", "TACSTD2", "ABCC11", "MKI67",
               "KRT23", "ALDH1A3", "SFRP1", "KRT15", "MYLK", "ACTA2", "GJB2", "SFRP4", "POSTN", "MMP2",
               "CXCR4", "CD8A", "TRAC", "CD4", "MS4A1", "BANK1", "APOC1", "MMP12", "C15orf48", "ITGAX", "CD68",
               "LRRC15", "AQP1", "VWF", "PECAM1", "CD3E", "EPCAM"]

# plot scatterplot of correlation values
plt.figure(figsize=(10, 5))
plt.scatter(visium_visimage_corr["Pearson"], xenium_xenimage_corr["Pearson"], alpha=1, c="black")
# plot line of 0,1
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("Pearson Correlation Values")
sns.despine()

# Highlight the selected genes in red and adjust text to avoid overlap
texts = []
for gene in genes_names:
    y = xenium_xenimage_corr.loc[xenium_xenimage_corr["Gene"] == gene, "Pearson"].values[0]
    x = visium_visimage_corr.loc[visium_visimage_corr["Gene"] == gene, "Pearson"].values[0]
    
    plt.scatter(x, y, color="#C83A27", s=50, label=gene if gene not in plt.gca().get_legend_handles_labels()[1] else "")
    texts.append(
        plt.text(
            x, y, gene, fontsize=10, color="#989B9E"
        )
    )

# Adjust text to avoid overlap
adjust_text(
    texts,
    x=visium_visimage_corr["Pearson"],
    y=xenium_xenimage_corr["Pearson"],
    arrowprops=dict(arrowstyle="-", color="#989B9E", lw=0.5),
    expand_points=(1.2, 1.2),
    expand_text=(1.2, 1.2),
    force_text=0.5,
    force_points=0.5
)

plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig11_scatterplot1.svg", dpi=1000, bbox_inches="tight")



# import matplotlib.colors as mcolors
# from adjustText import adjust_text

# # ------------------------------------------------------------
# # Genes to highlight
# # ------------------------------------------------------------
# genes_names = [
#     "FASN", "FOXA1", "CEACAM6", "GATA3", "MZB1", "AGR3", "SERPINA3",
#     "TACSTD2", "ABCC11", "MKI67", "KRT23", "ALDH1A3", "SFRP1", "KRT15",
#     "MYLK", "ACTA2", "GJB2", "SFRP4", "POSTN", "MMP2", "CXCR4", "CD8A",
#     "TRAC", "CD4", "MS4A1", "BANK1", "APOC1", "MMP12", "C15orf48",
#     "ITGAX", "CD68", "LRRC15", "AQP1", "VWF", "PECAM1", "CD3E", "EPCAM"
# ]

# # ------------------------------------------------------------
# # Base scatter: all genes
# # ------------------------------------------------------------
# plt.figure(figsize=(10, 5))

# plt.scatter(
#     visium_visimage_corr["Pearson"],
#     xenium_xenimage_corr["Pearson"],
#     c="black",
#     alpha=1,
#     zorder=1
# )

# # Diagonal reference line
# plt.plot([0, 1], [0, 1], linestyle="--", color="black", lw=2)

# plt.xlabel("Visium data – Visium Image")
# plt.ylabel("Xenium data – Xenium Image")
# plt.title("Pearson Correlation Values")
# sns.despine()

# # ------------------------------------------------------------
# # Collect highlighted gene data safely
# # ------------------------------------------------------------
# xs, ys, exprs, labels = [], [], [], []

# for gene in genes_names:
#     if (
#         gene in visium_visimage_corr["Gene"].values
#         and gene in xenium_xenimage_corr["Gene"].values
#         and gene in adata_xenium.var.index
#     ):
#         xs.append(
#             visium_visimage_corr.loc[
#                 visium_visimage_corr["Gene"] == gene, "Pearson"
#             ].iloc[0]
#         )
#         ys.append(
#             xenium_xenimage_corr.loc[
#                 xenium_xenimage_corr["Gene"] == gene, "Pearson"
#             ].iloc[0]
#         )
#         exprs.append(
#             adata_xenium.var.loc[gene, "avg_expression"]
#         )
#         labels.append(gene)

# xs = np.array(xs, dtype=float)
# ys = np.array(ys, dtype=float)
# exprs = np.array(exprs, dtype=float)

# # ------------------------------------------------------------
# # Highlight genes with expression-based coloring (FIXED)
# # ------------------------------------------------------------
# norm = mcolors.Normalize(
#     vmin=exprs.min(),
#     vmax=exprs.max()
# )

# sc = plt.scatter(
#     xs,
#     ys,
#     c=exprs,
#     cmap="viridis",
#     norm=norm,
#     s=60,
#     # edgecolor="black",
#     zorder=3
# )

# # ------------------------------------------------------------
# # Add labels and automatically adjust to avoid overlap
# # ------------------------------------------------------------
# texts = []
# for x, y, gene in zip(xs, ys, labels):
#     texts.append(
#         plt.text(
#             x,
#             y,
#             gene,
#             fontsize=10,
#             color="#989B9E"
#         )
#     )

# adjust_text(
#     texts,
#     x=xs,
#     y=ys,
#     arrowprops=dict(
#         arrowstyle="-",
#         color="#989B9E",
#         lw=1
#     ),
#     expand_points=(1.2, 1.2),
#     expand_text=(1.2, 1.2),
#     force_text=0.5,
#     force_points=0.5
# )

# # ------------------------------------------------------------
# # Colorbar
# # ------------------------------------------------------------
# cbar = plt.colorbar(sc)
# cbar.set_label("Average Gene Expression (Xenium)")

# plt.tight_layout()
# plt.show()


# plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig11_scatterplot1.svg", dpi=1000, bbox_inches="tight")



# rmse ##


genes_names = ["FASN", "FOXA1", "CEACAM6", "GATA3", "MZB1", "AGR3", "SERPINA3", "TACSTD2", "ABCC11", "MKI67",
               "KRT23", "ALDH1A3","SFRP1", "KRT15","MYLK","ACTA2","GJB2","SFRP4","POSTN","MMP2",
               "CXCR4","CD8A","TRAC","CD4","MS4A1","BANK1","APOC1","MMP12","C15orf48","ITGAX","CD68",
               "LRRC15","AQP1","VWF","PECAM1", "CD3E", "EPCAM"]

# plot scatterplot of correlation values
plt.figure(figsize=(10, 5))
plt.scatter(visium_visimage_corr["rMSE_range"], xenium_xenimage_corr["rMSE_range"], alpha=1, c="black")
# plot line of 0,1
plt.plot([0, .3], [0, .3], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data - Visium Image")
plt.ylabel("Xenium data - Xenium Image")
plt.title("rMSE_range")
sns.despine()

# Highlight the selected genes in green and adjust text to avoid overlap
texts = []
for gene in genes_names:
    y = xenium_xenimage_corr.loc[xenium_xenimage_corr["Gene"] == gene, "rMSE_range"].values[0]
    x = visium_visimage_corr.loc[visium_visimage_corr["Gene"] == gene, "rMSE_range"].values[0]
    
    plt.scatter(x, y, color="#C83A27", s=50, label=gene if gene not in plt.gca().get_legend_handles_labels()[1] else "")
    texts.append(
        plt.text(
            x, y, gene, fontsize=10, color="#989B9E"
        )
    )

# Adjust text to avoid overlap
adjust_text(
    texts,
    x=visium_visimage_corr["rMSE_range"],
    y=xenium_xenimage_corr["rMSE_range"],
    arrowprops=dict(arrowstyle="-", color="#989B9E", lw=0.5),
    expand_points=(1.2, 1.2),
    expand_text=(1.2, 1.2),
    force_text=0.5,
    force_points=0.5
)

plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig11_scatterplot2.svg", dpi=1000, bbox_inches="tight")





# ### plot only gene names ###



# # plot scatterplot of correlation values for selected genes only
# plt.figure(figsize=(10, 5))

# # Filter the data for the selected genes
# filtered_visium = visium_visimage_corr[visium_visimage_corr["Gene"].isin(genes_names)]
# filtered_xenium = xenium_xenimage_corr[xenium_xenimage_corr["Gene"].isin(genes_names)]

# # Plot only the selected genes
# plt.scatter(filtered_visium["Pearson"], filtered_xenium["Pearson"], alpha=1, c="black")

# # plot line of 0,1
# plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data w/ Visium Image")
# plt.ylabel("Xenium data w/ Xenium Image")
# plt.title("Pearson Correlation Values")
# sns.despine()

# # Annotate the selected genes on the scatter plot
# for gene in genes_names:
#     y = filtered_xenium.loc[filtered_xenium["Gene"] == gene, "Pearson"].values[0]
#     x = filtered_visium.loc[filtered_visium["Gene"] == gene, "Pearson"].values[0]
    
#     plt.annotate(
#         gene,
#         (x, y),  # Point coordinates
#         xytext=(x + 0.02, y + 0.02),  # Offset for the label
#         textcoords='offset points',  # Interpret `xytext` as offset in points
#         fontsize=12,
#         color="green"
#     )


# ### rmse plot ###


# genes_names = ["MS4A1", "CD3E", "EPCAM", "CEACAM6", "VWF", "POSTN", "CD68", "MZB1"]

# # plot scatterplot of correlation values for selected genes only
# plt.figure(figsize=(10, 5))

# # Filter the data for the selected genes
# filtered_visium = visium_visimage_corr[visium_visimage_corr["Gene"].isin(genes_names)]
# filtered_xenium = xenium_xenimage_corr[xenium_xenimage_corr["Gene"].isin(genes_names)]

# # Plot only the selected genes
# plt.scatter(filtered_visium["rMSE_range"], filtered_xenium["rMSE_range"], alpha=1, c="black")

# # plot line of 0,1
# plt.plot([0, .25], [0, .25], color="black", linestyle="--", lw=2)
# plt.xlabel("Visium data w/ Visium Image")
# plt.ylabel("Xenium data w/ Xenium Image")
# plt.title("rMSE_range Correlation Values")
# sns.despine()

# # Annotate the selected genes on the scatter plot
# for gene in genes_names:
#     y = filtered_xenium.loc[filtered_xenium["Gene"] == gene, "rMSE_range"].values[0]
#     x = filtered_visium.loc[filtered_visium["Gene"] == gene, "rMSE_range"].values[0]
    
#     plt.annotate(
#         gene,
#         (x, y),  # Point coordinates
#         xytext=(x + 0.02, y + 0.02),  # Offset for the label
#         textcoords='offset points',  # Interpret `xytext` as offset in points
#         fontsize=12,
#         color="green"
#     )




########################################################################################################################


### pearson vs expression magnitude ###



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


import scanpy as sc
import scipy.sparse


# read in adatas
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# log the expression
# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)


# Calculate average gene expression for each gene
adata_xenium.var["avg_expression"] = adata_xenium.X.mean(axis=0).A1
adata_visium.var["avg_expression"] = adata_visium.X.mean(axis=0).A1

# Merge average expression with Pearson correlation data
xenium_corr = xenium_xenimage_corr.merge(adata_xenium.var[["avg_expression"]], left_on="Gene", right_index=True)
visium_corr = visium_visimage_corr.merge(adata_visium.var[["avg_expression"]], left_on="Gene", right_index=True)

# Plot scatterplot for Xenium data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["avg_expression"], y=xenium_corr["Pearson"], color="#E69F01", label="Xenium")
plt.xlabel("Average Gene Expression")
plt.ylabel("Pearson Correlation")
plt.title("Xenium: Average Gene Expression vs Pearson Correlation")
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot1.svg", dpi=1000, bbox_inches="tight")



# # make a scatterplot of these genes

# genes_names = ["FASN", "FOXA1", "CEACAM6", "GATA3", "MZB1", "AGR3", "SERPINA3", "TACSTD2", "ABCC11", "MKI67",
#                "KRT23", "ALDH1A3","SFRP1", "KRT15","MYLK","ACTA2","GJB2","SFRP4","POSTN","MMP2",
#                "CXCR4","CD8A","TRAC","CD4","MS4A1","BANK1","APOC1","MMP12","C15orf48","ITGAX","CD68",
#                "LRRC15","AQP1","VWF","PECAM1", "CD3E", "EPCAM"]

# # Plot scatterplot for Xenium data
# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=xenium_corr["avg_expression"], y=xenium_corr["Pearson"], color="#E69F01", label="Xenium")

# # Highlight the selected genes in different colors
# for i, gene in enumerate(genes_names):
#     y = xenium_corr.loc[xenium_corr["Gene"] == gene, "Pearson"].values[0]
#     x = xenium_corr.loc[xenium_corr["Gene"] == gene, "avg_expression"].values[0]
    
#     plt.scatter(x, y, label=gene, s=50, alpha=0.8)
#     plt.annotate(
#         gene,
#         (x, y),  # Point coordinates
#         xytext=(10, 10),  # Offset for the label
#         textcoords='offset points',  # Interpret `xytext` as offset in points
#         fontsize=8,
#         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
#     )

# plt.xlabel("Average Gene Expression")
# plt.ylabel("Pearson Correlation")
# plt.title("Xenium: Average Gene Expression vs Pearson Correlation")
# sns.despine()
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.tight_layout()
# # plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot1.svg", dpi=1000, bbox_inches="tight")




# Plot scatterplot for Visium data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_corr["avg_expression"], y=visium_corr["Pearson"], color="#55B4E9", label="Visium")
plt.xlabel("Average Gene Expression")
plt.ylabel("Pearson Correlation")
plt.title("Visium: Average Gene Expression vs Pearson Correlation")
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot2.svg", dpi=1000, bbox_inches="tight")


# Find genes with highest gene expression but lowest Pearson correlation
xenium_corr["expression_to_correlation"] = xenium_corr["avg_expression"] / (xenium_corr["Pearson"] + 1e-6)
visium_corr["expression_to_correlation"] = visium_corr["avg_expression"] / (visium_corr["Pearson"] + 1e-6)

# Sort by the expression_to_correlation ratio
xenium_outliers = xenium_corr.sort_values("expression_to_correlation", ascending=False).head(10)
visium_outliers = visium_corr.sort_values("expression_to_correlation", ascending=False).head(10)

# Print the results
print("Xenium outliers:")
print(xenium_outliers[["Gene", "avg_expression", "Pearson"]])

print("\nVisium outliers:")
print(visium_outliers[["Gene", "avg_expression", "Pearson"]])



### plot coad results for this ###



# read in data #

visiumhd_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed42_test_correlation_df.csv", index_col=0)
visiumhd_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed1_test_correlation_df.csv", index_col=0)
visiumhd_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed10_test_correlation_df.csv", index_col=0)
visiumhd_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed100_test_correlation_df.csv", index_col=0)
visiumhd_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/visiumhd_patchsize210_seed0_test_correlation_df.csv", index_col=0)
visiumhd_corr = pd.concat([visiumhd_corr42, visiumhd_corr0, visiumhd_corr1, visiumhd_corr10, visiumhd_corr100])
visiumhd_corr_all = visiumhd_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
visiumhd_corr = visiumhd_corr_all.groupby("Gene").mean().reset_index()


xenium_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed42_test_correlation_df.csv", index_col=0)
xenium_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed1_test_correlation_df.csv", index_col=0)
xenium_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed10_test_correlation_df.csv", index_col=0)
xenium_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed100_test_correlation_df.csv", index_col=0)
xenium_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/xenium_patchsize250_seed0_test_correlation_df.csv", index_col=0)
xenium_corr = pd.concat([xenium_corr42, xenium_corr0, xenium_corr1, xenium_corr10, xenium_corr100])
xenium_corr_all = xenium_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
xenium_corr = xenium_corr_all.groupby("Gene").mean().reset_index()


cosmx_corr42 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed42_test_correlation_df.csv", index_col=0)
cosmx_corr1 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed1_test_correlation_df.csv", index_col=0)
cosmx_corr10 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed10_test_correlation_df.csv", index_col=0)
cosmx_corr100 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed100_test_correlation_df.csv", index_col=0)
cosmx_corr0 = pd.read_csv("/home/caleb/Desktop/improvedgenepred/07_revision/results/cosmx_patchsize250_seed0_test_correlation_df.csv", index_col=0)
cosmx_corr = pd.concat([cosmx_corr42, cosmx_corr0, cosmx_corr1, cosmx_corr10, cosmx_corr100])
cosmx_corr_all = cosmx_corr.sort_values("Gene")
# group by gene and take the mean of the Pearson correlation
cosmx_corr = cosmx_corr_all.groupby("Gene").mean().reset_index()




# read in dataasets 

patch_size = 250 # 250 or 210
# combined data
method = 'cosmx' # cosmx xenium visiumhd
adata_cosmx = sc.read_h5ad(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_data_{patch_size}.h5ad')
method = 'xenium' # cosmx xenium visiumhd
adata_xen= sc.read_h5ad(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_data_{patch_size}.h5ad')
method = 'visiumhd' # cosmx xenium visiumhd
patch_size = 210 # 250 or 210
adata_vis = sc.read_h5ad(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_data_{patch_size}.h5ad')


# make .X a csr matrix
adata_cosmx.X = scipy.sparse.csr_matrix(adata_cosmx.X)
adata_vis.X = scipy.sparse.csr_matrix(adata_vis.X)
adata_xen.X = scipy.sparse.csr_matrix(adata_xen.X)

# Calculate average gene expression for each gene
adata_cosmx.var["avg_expression"] = adata_cosmx.X.mean(axis=0).A1
adata_vis.var["avg_expression"] = adata_vis.X.mean(axis=0).A1
adata_xen.var["avg_expression"] = adata_xen.X.mean(axis=0).A1

# Merge average expression with Pearson correlation data
cosmx_corr = cosmx_corr.merge(adata_cosmx.var[["avg_expression"]], left_on="Gene", right_index=True)
visium_corr = visiumhd_corr.merge(adata_vis.var[["avg_expression"]], left_on="Gene", right_index=True)
xenium_corr = xenium_corr.merge(adata_xen.var[["avg_expression"]], left_on="Gene", right_index=True)

# Plot scatterplot for CosMx data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=cosmx_corr["avg_expression"], y=cosmx_corr["Pearson"], color="C6", label="CosMx")
plt.xlabel("Average Gene Expression")
plt.ylabel("Pearson Correlation")
plt.title("CosMx: Average Gene Expression vs Pearson Correlation")
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot3.svg", dpi=1000, bbox_inches="tight")


# Plot scatterplot for VisiumHD data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_corr["avg_expression"], y=visium_corr["Pearson"], color="#55B4E9", label="VisiumHD")
plt.xlabel("Average Gene Expression")
plt.ylabel("Pearson Correlation")
plt.title("VisiumHD: Average Gene Expression vs Pearson Correlation")
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot4.svg", dpi=1000, bbox_inches="tight")


# Plot scatterplot for Xenium data
plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_corr["avg_expression"], y=xenium_corr["Pearson"], color="#E69F01", label="Xenium")
plt.xlabel("Average Gene Expression")
plt.ylabel("Pearson Correlation")
plt.title("Xenium: Average Gene Expression vs Pearson Correlation")
sns.despine()
plt.savefig("/home/caleb/Desktop/improvedgenepred/07_revision/figures/suppfig12_scatterplot5.svg", dpi=1000, bbox_inches="tight")




########################################################################################################################
