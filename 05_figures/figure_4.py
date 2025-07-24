### read in data and make tables ###

import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns


##############################################################################################################


# function to make table
def make_table(file_names, base_path):
    # Initialize an empty DataFrame
    data = pd.DataFrame()
    
    # Loop through each file
    for file in file_names:
        # Read the file data
        data_tmp = pd.read_csv(f'{base_path}/{file}', sep='\t', header=None)
        
        # Extract spearman, pearson, and mse values using regular expressions
        spearman = round(float(re.search(r'correlation: (.+)', data_tmp.iloc[0,0])[1]), 4)
        pearson = round(float(re.search(r'correlation: (.+)', data_tmp.iloc[1,0])[1]), 4)
        mse = round(float(re.search(r'MSE: (.+)', data_tmp.iloc[2,0])[1]), 4)

        # Extract data name
        # data_name = re.search(r'breastcancer_(.+)_res', file)[1] + '_' + re.search(r'([^_]+)\.txt$', file)[1]
        data_name = re.search(r'([^_]+)\.txt$', file)[1]

        # Append the extracted values into the DataFrame
        data = pd.concat([data, pd.DataFrame({'spearman': spearman, 'pearson': pearson, 'mse': mse}, index=[data_name])])
    
    # Sort by pearson correlation in descending order
    # data = data.sort_values(by='pearson', ascending=False)
    data = data.sort_index()
    
    return data



##############################################################################################################

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


##############################################################################################################


# visium vs xenium on visium image #

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_visimage_corr["Pearson"], color="#55B4E9", label="Visium data w/ Visium Image", kde=True)
sns.histplot(visium_xenimage_corr["Pearson"], color="C4", label="Visium data w/ Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(visium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="#55B4E9", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="#E69F01", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="C4", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="C4", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("fig4a_histogram.svg", dpi=1000, bbox_inches="tight")


##############################################################################################################

# visium vs xenium on xenium image #

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(xenium_visimage_corr["Pearson"], color="C3", label="Xenium data w/ Visium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data w/ Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_xenimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="#55B4E9", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="#E69F01", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="C3", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="C3", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="#E69F01", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("fig4a_histogram2.svg", dpi=1000, bbox_inches="tight")



# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=xenium_visimage_corr["Pearson"], y=xenium_xenimage_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Xenium data w/ Visium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig4b_scatterplot.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["Pearson"], y=visium_xenimage_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Visium data w/ Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig4b_scatterplot2.svg", dpi=1000, bbox_inches="tight")





##############################################################################################################

### plots for image quality ###


# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# get xenium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and "rep2" not in file and "xeniumdata" in file] 
file_names_xenium.append("breastcancer_visiumdata_visiumimage_seed42_test_correlation_summary_none.txt")

# get the file names for the full data
# file_names_xenium_full = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' in file]
# get the file names for rep2
file_names_xenium_rep2 = [file for file in file_names if 'xenium' in file and 'summary' in file and 'rep2' in file and "blur" not in file and "xeniumdata" in file]
file_names_xenium_rep2.append("breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_summary_none.txt")


# get the file names for rep2 with blur
file_names_xenium_rep2_blur = [file for file in file_names if 'xenium' in file and 'summary' in file and 'rep2' in file and "blur" in file and "xeniumdata" in file]
file_names_xenium_rep2_blur.append("breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_summary_none.txt")

# add visium
# file_names_xenium_rep2.append("breastcancer_visium_imagexenium_seed42_correlation_summary_rep2_visium.txt")

# index to sort by
desired_order = ['xenium', 'gb5', 'gb9', 'gb15', 'gb21', 'gb25', 'visium']


# make tables
table_xenium = make_table(file_names_xenium, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_xenium.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium", "visium"]
# sort manually
table_xenium = table_xenium.reindex(desired_order)
table_xenium

table_xenium_rep2 = make_table(file_names_xenium_rep2, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_xenium_rep2.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium", "visium"]
table_xenium_rep2 = table_xenium_rep2.reindex(desired_order)
table_xenium_rep2

table_xenium_rep2_blur = make_table(file_names_xenium_rep2_blur, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_xenium_rep2_blur.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium"]
table_xenium_rep2_blur = table_xenium_rep2_blur.reindex(['xenium', 'gb5', 'gb9', 'gb15', 'gb21', 'gb25'])
table_xenium_rep2_blur


# plot
# Plot the line excluding the last point
plt.figure(figsize=(10, 5))
sns.lineplot(x=table_xenium.index[:-1], y='mse', data=table_xenium.iloc[:-1], label='Test', marker='o', markersize=10)
sns.lineplot(x=table_xenium_rep2.index[:-1], y='mse', data=table_xenium_rep2.iloc[:-1], label='Rep2', marker='o', markersize=10)
sns.lineplot(x=table_xenium_rep2_blur.index, y='mse', data=table_xenium_rep2_blur, label='Rep2 Blur', marker='o', markersize=10)

# Plot the last point separately as a dot
plt.scatter(table_xenium.index[-1], table_xenium['mse'].iloc[-1], color='C0', zorder=5)
plt.scatter(table_xenium_rep2.index[-1], table_xenium_rep2['mse'].iloc[-1], color='C1', zorder=5)
# plt.scatter(table_xenium_rep2_blur.index[-1], table_xenium_rep2_blur['mse'].iloc[-1], color='green', label='Rep2 Blur Last Point', zorder=5)

# Customize the plot
plt.grid(axis='y')
plt.xlabel('Gaussian Blur on Image')
plt.ylabel('MSE')
plt.title('Effect of Increasing Gaussian Blur')
plt.legend()
plt.show()



# plot
# Plot the line excluding the last point
plt.figure(figsize=(10, 5))
sns.lineplot(x=table_xenium.index[:-1], y='pearson', data=table_xenium.iloc[:-1], label='Test', marker='o', markersize=10)
sns.lineplot(x=table_xenium_rep2.index[:-1], y='pearson', data=table_xenium_rep2.iloc[:-1], label='Rep2', marker='o', markersize=10)
sns.lineplot(x=table_xenium_rep2_blur.index, y='pearson', data=table_xenium_rep2_blur, label='Rep2 Blur', marker='o', markersize=10)

# Plot the last point separately as a dot
plt.scatter(table_xenium.index[-1], table_xenium['pearson'].iloc[-1], color='C0', zorder=5, s=75)
plt.scatter(table_xenium_rep2.index[-1], table_xenium_rep2['pearson'].iloc[-1], color='C1', zorder=5, s=75)
# plt.scatter(table_xenium_rep2_blur.index[-1], table_xenium_rep2_blur['mse'].iloc[-1], color='green', label='Rep2 Blur Last Point', zorder=5)

# Customize the plot
plt.grid(axis='y')
plt.xlabel('Gaussian Blur on Image')
plt.ylabel('Pearson Correlation')
plt.title('Effect of Increasing Gaussian Blur')
plt.legend()
# plt.savefig("fig4c_gaussianblur.svg", dpi=1000, bbox_inches="tight")
# plt.show()




### add visium data ###

# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# get xenium file names
file_names_visium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and "rep2" not in file and "visiumdata" in file] 
file_names_visium.append("breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_summary_none.txt")
file_names_visium.append("breastcancer_visiumdata_visiumimage_seed42_test_correlation_summary_none.txt")

# get the file names for the full data
# file_names_xenium_full = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' in file]
# get the file names for rep2
file_names_visium_rep2 = [file for file in file_names if 'xenium' in file and 'summary' in file and 'rep2' in file and "blur" not in file and "visiumdata" in file]
file_names_visium_rep2.append("breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_summary_none.txt")
file_names_visium_rep2.append("breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_summary_none.txt")


# get the file names for rep2 with blur
file_names_visium_rep2_blur = [file for file in file_names if 'xenium' in file and 'summary' in file and 'rep2' in file and "blur" in file and "visiumdata" in file]
file_names_visium_rep2_blur.append("breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_summary_none.txt")


# index to sort by
desired_order = ['xenium', 'gb5', 'gb9', 'gb15', 'gb21', 'gb25', 'visium']


# make tables
table_visium = make_table(file_names_visium, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_visium.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium", "visium"]
# sort manually
table_visium = table_visium.reindex(desired_order)
table_visium

table_visium_rep2 = make_table(file_names_visium_rep2, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_visium_rep2.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium", "visium"]
table_visium_rep2 = table_visium_rep2.reindex(desired_order)
table_visium_rep2

table_visium_rep2_blur = make_table(file_names_visium_rep2_blur, '/home/caleb/Desktop/improvedgenepred/results/img_quality/')
# reindex
table_visium_rep2_blur.index = ["gb15", "gb21", "gb25", "gb5", "gb9", "xenium"]
table_visium_rep2_blur = table_visium_rep2_blur.reindex(['xenium', 'gb5', 'gb9', 'gb15', 'gb21', 'gb25'])
table_visium_rep2_blur


# plot
# Plot the line excluding the last point
plt.figure(figsize=(10, 5))
sns.lineplot(x=table_visium.index[:-1], y='pearson', data=table_visium.iloc[:-1], label='Test', marker='o', markersize=10)
sns.lineplot(x=table_visium_rep2.index[:-1], y='pearson', data=table_visium_rep2.iloc[:-1], label='Rep2', marker='o', markersize=10)
sns.lineplot(x=table_visium_rep2_blur.index, y='pearson', data=table_visium_rep2_blur, label='Rep2 Blur', marker='o', markersize=10)

# Plot the last point separately as a dot
plt.scatter(table_visium.index[-1], table_visium['pearson'].iloc[-1], color='C0', zorder=5)
plt.scatter(table_visium_rep2.index[-1], table_visium_rep2['pearson'].iloc[-1], color='C1', zorder=5)
# plt.scatter(table_visium_rep2_blur.index[-1], table_visium_rep2_blur['pearson'].iloc[-1], color='green', label='Rep2 Blur Last Point', zorder=5)

# Customize the plot
plt.grid(axis='y')
plt.xlabel('Gaussian Blur on Image')
plt.ylabel('Pearson Correlation')
plt.title('Effect of Increasing Gaussian Blur - Visium')
plt.legend()
plt.show()








### NEW IMAGE Quality ANALYSIS ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define seeds, sparsity thresholds, and replicates
seeds = [42, 0, 1, 10, 100]
conditions = ['Xenium'] + [5, 25, 125] + ['Visium']
replicates = ['test', 'rep2', 'rep2_blurred']

# Base path templates
sparsity_path = "/home/caleb/Desktop/improvedgenepred/results/img_quality/" \
                "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_gb{cond}.csv"
xenium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_none.csv"
visium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_none.csv"

# Function to compute mean Pearson per seed for a given file pattern
def compute_seed_means(path_template, cond_key, rep):
    means = []
    for seed in seeds:
        path = path_template.format(seed=seed, rep=rep, cond=cond_key)
        print(f"Loading for rep={rep!r}, seed={seed}, cond={cond_key!r}: {path}")
        df = pd.read_csv(path, index_col=0)
        means.append(df['Pearson'].mean())
    return means

# Collect results
results = {rep: {'means': [], 'sems': []} for rep in replicates}

for rep in replicates:
    all_means = []
    all_sems = []
    # Xenium baseline (skip for rep2_blurred)
    if rep != 'rep2_blurred':
        xm = compute_seed_means(xenium_path, None, rep)
        all_means.append(np.mean(xm))
        all_sems.append(np.std(xm, ddof=1) / np.sqrt(len(xm)))
    # Sparsity thresholds (both reps)
    for cond in [5, 25, 125]:
        sp = compute_seed_means(sparsity_path, cond, rep)
        all_means.append(np.mean(sp))
        all_sems.append(np.std(sp, ddof=1) / np.sqrt(len(sp)))

    if rep != 'rep2_blurred':
        vis = compute_seed_means(visium_path, None, rep)
        all_means.append(np.mean(vis))
        all_sems.append(np.std(vis, ddof=1) / np.sqrt(len(vis)))
    results[rep]['means'] = all_means
    results[rep]['sems'] = all_sems

# Convert conditions for labels
conditions = ['Xenium', 'gb5', 'gb25', 'gb125', 'Visium']
x = np.arange(len(conditions))

fig, ax = plt.subplots(figsize=(8, 5))

for idx, rep in enumerate(replicates):
    y = results[rep]['means']
    yerr = results[rep]['sems']
    color = f'C{idx}'  # C0, C1, C2
    if rep == 'rep2_blurred':
        # Only plot at sparsity conditions positions 1 and 2
        # Plot error bars first (zorder=1), then points (zorder=2)
        ax.errorbar(x[1:4], y, yerr=yerr, fmt='none', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8, zorder=1)
        ax.plot(x[1:4], y, '--o', color=color, label=rep, zorder=2, linewidth=0.5, markersize=8)
    elif rep == 'rep2':
        # Plot only at sparsity conditions positions 1 and 2
        ax.errorbar(x[[0,4]], [y[0], y[4]], yerr=[yerr[0], yerr[4]], fmt='none', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8, zorder=1)
        ax.plot(x[[0,4]], [y[0], y[4]], 'o', color=color, label=rep, zorder=2, linewidth=0.5, markersize=8)
    else:
        # Plot full series: connect all except final Visium point
        ax.errorbar(x[:-1], y[:-1], yerr=yerr[:-1], fmt='none', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8, zorder=1)
        ax.plot(x[:-1], y[:-1], '--o', color=color, label=rep, zorder=2, linewidth=0.5, markersize=8)
        # Plot the final Visium point without connecting line
        ax.errorbar(x[-1], y[-1], yerr=yerr[-1], fmt='none', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8, zorder=1)
        ax.plot(x[-1], y[-1], 'o', color=color, zorder=2, linewidth=0.5, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Mean Pearson Correlation')
ax.set_title('Pearson vs. Condition across Replicates')
ax.legend(title='Replicate')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("fig4c_gaussianblur.svg", dpi=1000, bbox_inches="tight")






