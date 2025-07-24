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
sns.histplot(xenium_visimage_corr["Pearson"], color="C3", label="Xenium data w/ Visium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_visimage_corr["Pearson"])
visium_mean = np.mean(visium_visimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="#55B4E9", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="#E69F01", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="#55B4E9", linestyle="--")
plt.axvline(xenium_mean, color="C3", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="#55B4E9", ha="center")
plt.text(xenium_mean, plt.ylim()[1]*1, f"{np.round(xenium_mean,3)}", 
         fontsize=10, color="C3", ha="center")

plt.xlabel("Pearson Correlation")
plt.ylabel("Frequency")
# plt.title("Correlation Distribution")
# get rid of the top and right spines
sns.despine()
# make x axis start at 0 and end at 1
plt.xlim(0, 1)
plt.legend()
# save the plot
plt.savefig("fig3a_histogram.svg", dpi=1000, bbox_inches="tight")


##############################################################################################################

# visium vs xenium on xenium image #

# plot correlation distributions
plt.figure(figsize=(10, 5))
sns.histplot(visium_xenimage_corr["Pearson"], color="C4", label="Visium data w/ Xenium Image", kde=True)
sns.histplot(xenium_xenimage_corr["Pearson"], color="#E69F01", label="Xenium data w/ Xenium Image", kde=True)

# plot average correlation
xenium_mean = np.mean(xenium_xenimage_corr["Pearson"])
visium_mean = np.mean(visium_xenimage_corr["Pearson"])
# plt.axvline(xenium_mean, color="#55B4E9", linestyle="--", label="Mean Xenium w/ Visium Image")
# plt.axvline(visium_mean, color="#E69F01", linestyle="--", label="Mean Visium w/ Visium Image")
plt.axvline(visium_mean, color="C4", linestyle="--")
plt.axvline(xenium_mean, color="#E69F01", linestyle="--")

# plot the average correlation of the two datasets on the plot in text at the top of each line
plt.text(visium_mean, plt.ylim()[1]*1, f"{np.round(visium_mean, 3)}", 
         fontsize=10, color="C4", ha="center")
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
plt.savefig("fig3a_histogram2.svg", dpi=1000, bbox_inches="tight")



# scatterplot of xenium vs visium on visium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_visimage_corr["Pearson"], y=xenium_visimage_corr["Pearson"], c="black", linewidth = 0)
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Visium Image")
plt.ylabel("Xenium data w/ Visium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig3b_scatterplot.svg", dpi=1000, bbox_inches="tight")


# scatterplot of xenium vs visium on xenium image #

plt.figure(figsize=(10, 5))
sns.scatterplot(x=visium_xenimage_corr["Pearson"], y=xenium_xenimage_corr["Pearson"], alpha=1, c="black",linewidth = 0)
plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2)
plt.xlabel("Visium data w/ Xenium Image")
plt.ylabel("Xenium data w/ Xenium Image")
plt.title("Pearson Correlation Values")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
sns.despine()
plt.savefig("fig3b_scatterplot2.svg", dpi=1000, bbox_inches="tight")


##############################################################################################################


### sparsity analysis ###

# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/sparsity')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and 'test' in file] 
file_names_xenium.append('breastcancer_visiumdata_visiumimage_seed42_test_correlation_summary_none.txt')

# make tables
table_xenium = make_table(file_names_xenium, '/home/caleb/Desktop/improvedgenepred/results/sparsity')
table_xenium
# reindex
table_xenium.index = ['<=0', '<=1', '<=10', '<=15', '<=20', '<=5', 'xenium', 'visium']

# reorder the index
desired_order = ['xenium', '<=0', '<=1', '<=5', '<=10', '<=15', '<=20', 'visium']
table_xenium = table_xenium.reindex(desired_order)
table_xenium


# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/sparsity')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and 'rep2' in file] 
file_names_xenium.append('breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_summary_none.txt')


# make tables
table_xenium_rep2 = make_table(file_names_xenium, '/home/caleb/Desktop/improvedgenepred/results/sparsity')
table_xenium_rep2
# reindex
table_xenium_rep2.index = ['<=0', '<=1', '<=10', '<=15', '<=20', '<=5', 'xenium', 'visium']

# reorder the index
desired_order = ['xenium', '<=0', '<=1', '<=5', '<=10', '<=15', '<=20', 'visium']
table_xenium_rep2 = table_xenium_rep2.reindex(desired_order)



# Plotting for table_xenium
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=table_xenium.index[:-1], y=table_xenium['pearson'][:-1], 
    label='Test Set', marker='o', markersize=10
)
plt.scatter(
    table_xenium.index[-1], table_xenium['pearson'].iloc[-1], 
    zorder=3, s=75
)

# Plotting for table_xenium_rep2
sns.lineplot(
    x=table_xenium_rep2.index[:-1], y=table_xenium_rep2['pearson'][:-1], 
    label='Rep2', marker='o', markersize=10
)
plt.scatter(
    table_xenium_rep2.index[-1], table_xenium_rep2['pearson'].iloc[-1], 
    zorder=3, s=75
)


# Customizing the plot
plt.grid(axis='y')
plt.xlabel('Method')
plt.ylabel('Pearson Correlation')
plt.title('Xenium Data Xenium Image')
plt.legend()
# plt.savefig("fig3c_sparsity.svg", dpi=1000, bbox_inches="tight")




# Plotting for table_xenium
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=table_xenium.index[:-1], y=table_xenium['mse'][:-1], 
    label='Test Set', marker='o', markersize=10
)
plt.scatter(
    table_xenium.index[-1], table_xenium['mse'].iloc[-1], 
    zorder=3, s=75
)

# Plotting for table_xenium_rep2
sns.lineplot(
    x=table_xenium_rep2.index[:-1], y=table_xenium_rep2['mse'][:-1], 
    label='Rep2', marker='o', markersize=10
)
plt.scatter(
    table_xenium_rep2.index[-1], table_xenium_rep2['mse'].iloc[-1], 
    zorder=3, s=75
)

# Customizing the plot
plt.grid(axis='y')
plt.xlabel('Method')
plt.ylabel('MSE')
plt.title('Xenium Data Xenium Image')
plt.legend()
# plt.savefig("fig3c_sparsity.svg", dpi=1000, bbox_inches="tight")




### NEW SPARSITY ANALYSIS ###

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Define seeds, sparsity thresholds, and replicates
# seeds = [42, 0, 1, 10, 100]
# conditions = ['Xenium'] + [0, 1, 5, 10, 15, 20] + ['Visium']
# replicates = ['test', 'rep2']

# # Base path templates
# sparsity_path = "/home/caleb/Desktop/improvedgenepred/results/sparsity/" \
#                 "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_sparse_lessthanequalto{cond}.csv"
# xenium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
#                "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_none.csv"
# visium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
#                "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_none.csv"

# # Function to compute mean Pearson per seed for a given file pattern
# def compute_seed_means(path_template, cond_key, rep):
#     means = []
#     for seed in seeds:
#         path = path_template.format(seed=seed, rep=rep, cond=cond_key)
#         df = pd.read_csv(path, index_col=0)
#         valid_rows = df.loc[df['Pearson'].notna()]
#         gene_names = valid_rows['Gene']
#         print(len(gene_names), "genes with valid Pearson values for seed", seed, "and condition", cond_key)
#         print(gene_names)
#         means.append(df['Pearson'].mean())
#     return means

# # Collect results
# results = {rep: {'means': [], 'sems': []} for rep in replicates}

# for rep in replicates:
#     all_means = []
#     all_sems = []
#     # Xenium baseline
#     xm_means = compute_seed_means(xenium_path, None, rep)
#     all_means.append(np.mean(xm_means))
#     all_sems.append(np.std(xm_means, ddof=1) / np.sqrt(len(xm_means)))
#     # Sparsity thresholds
#     for cond in [0, 1, 5, 10, 15, 20]:
#         sp_means = compute_seed_means(sparsity_path, cond, rep)
#         all_means.append(np.mean(sp_means))
#         all_sems.append(np.std(sp_means, ddof=1) / np.sqrt(len(sp_means)))
#     # Visium baseline
#     vis_means = compute_seed_means(visium_path, None, rep)
#     all_means.append(np.mean(vis_means))
#     all_sems.append(np.std(vis_means, ddof=1) / np.sqrt(len(vis_means)))
    
#     results[rep]['means'] = all_means
#     results[rep]['sems'] = all_sems



import pandas as pd
import numpy as np

# --- parameters ---
seeds = [42, 0, 1, 10, 100]
conds = ['Xenium'] + [0,1,5,10,15,20] + ['Visium']
reps = ['test','rep2']

# Base path templates
sparsity_path = "/home/caleb/Desktop/improvedgenepred/results/sparsity/" \
                "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_sparse_lessthanequalto{cond}.csv"
xenium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_none.csv"
visium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_none.csv"

def _load_df(path):
    df = pd.read_csv(path, index_col=0)
    # only rows where Pearson is not NaN
    return df.loc[df['Pearson'].notna()]

# 1) Build a dict of gene-lists from 'test'
test_gene_lists = { cond: {} for cond in conds }

for cond in conds:
    template = (xenium_path if cond=='Xenium' 
                else visium_path if cond=='Visium' 
                else sparsity_path)
    for seed in seeds:
        path = template.format(seed=seed, rep='test', cond=cond)
        df_valid = _load_df(path)
        test_gene_lists[cond][seed] = set(df_valid['Gene'].tolist())

# 2) Compute means/sems for both replicates, filtering rep2 by test lists
results = { rep: {'means':[], 'sems':[]} for rep in reps }

for rep in reps:
    for cond in conds:
        template = (xenium_path if cond=='Xenium' 
                    else visium_path if cond=='Visium' 
                    else sparsity_path)
        seed_means = []
        for seed in seeds:
            path = template.format(seed=seed, rep=rep, cond=cond)
            df = _load_df(path)

            if rep == 'rep2':
                # filter to exactly those genes used in test
                mask = df['Gene'].isin(test_gene_lists[cond][seed])
                df = df.loc[mask]
                print(f"Rep2 {cond} seed {seed}: {len(df)} genes after filtering")

            seed_means.append(df['Pearson'].mean())

        # aggregate across seeds
        m = np.mean(seed_means)
        sem = np.std(seed_means, ddof=1) / np.sqrt(len(seed_means))
        results[rep]['means'].append(m)
        results[rep]['sems'].append(sem)

# now `results['test']` and `results['rep2']` both use the same gene-sets per seed
print(results)



conditions = ['Xenium'] + [f'<={cond}' for cond in [0, 1, 5, 10, 15, 20]] + ['Visium']

x = np.arange(len(conditions))
fig, ax = plt.subplots(figsize=(8, 5))

# Plot 'test' in color C0, 'rep2' in color C1
for idx, rep in enumerate(reps):
    y = results[rep]['means']
    yerr = results[rep]['sems']
    color = f'C{idx}'  # C0 for test, C1 for rep2
    # Connect all except final Visium point
    ax.errorbar(x[:-1], y[:-1], yerr=yerr[:-1], fmt='--o', color=color, label=f'{rep}', barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8)
    # Plot final Visium point without line
    ax.errorbar(x[-1], y[-1], yerr=yerr[-1], fmt='o', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Mean Pearson Correlation')
ax.set_title('Pearson vs. Condition across Replicates')
ax.legend(title='Replicate')
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("fig3c_sparsity.svg", dpi=1000, bbox_inches="tight")













##############################################################################################################


### poisson analysis ###

# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/poisson_noise')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and 'test' in file] 
file_names_xenium.append('breastcancer_visiumdata_visiumimage_seed42_test_correlation_summary_none.txt')

# make tables
table_xenium = make_table(file_names_xenium, '/home/caleb/Desktop/improvedgenepred/results/poisson_noise')
table_xenium
# reindex
table_xenium.index = ['xenium', 'visium', 'L1', 'L10', 'L100', 'L15', 'L20', 'L25', 'L5', 'L50', 'L75']

# reorder the index
desired_order = ['xenium', 'L1', 'L5', 'L10', 'L15', 'L20', 'L25', 'L50', 'L75', 'L100', 'visium']
table_xenium = table_xenium.reindex(desired_order)
table_xenium


# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/poisson_noise')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'summary' in file and 'full' not in file and 'rep2' in file] 
file_names_xenium.append('breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_summary_none.txt')


# make tables
table_xenium_rep2 = make_table(file_names_xenium, '/home/caleb/Desktop/improvedgenepred/results/poisson_noise')
table_xenium_rep2
# reindex
table_xenium_rep2.index = ['xenium', 'visium', 'L1', 'L10', 'L100', 'L15', 'L20', 'L25', 'L5', 'L50', 'L75']

# reorder the index
desired_order = ['xenium', 'L1', 'L5', 'L10', 'L15', 'L20', 'L25', 'L50', 'L75', 'L100', 'visium']
table_xenium_rep2 = table_xenium_rep2.reindex(desired_order)



# Plotting for table_xenium
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=table_xenium.index[:-1], y=table_xenium['pearson'][:-1], 
    label='Test Set', marker='o', markersize=10
)
plt.scatter(
    table_xenium.index[-1], table_xenium['pearson'].iloc[-1], 
    zorder=3, s=75
)

# Plotting for table_xenium_rep2
sns.lineplot(
    x=table_xenium_rep2.index[:-1], y=table_xenium_rep2['pearson'][:-1], 
    label='Rep2', marker='o', markersize=10
)
plt.scatter(
    table_xenium_rep2.index[-1], table_xenium_rep2['pearson'].iloc[-1], 
    zorder=3, s=75
)

# Customizing the plot
plt.grid(axis='y')
plt.xlabel('Method')
plt.ylabel('Pearson Correlation')
plt.title('Xenium Data Xenium Image')
plt.legend()
plt.savefig("fig3d_poisson.svg", dpi=1000, bbox_inches="tight")





# NEW poisson analysis #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define seeds,  thresholds, and replicates
seeds = [42, 0, 1, 10, 100]
conditions = ['Xenium'] + [5, 15, 45] + ['Visium']
replicates = ['test', 'rep2']

# Base path templates
sparsity_path = "/home/caleb/Desktop/improvedgenepred/results/poisson_noise/" \
                "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_poissonlamda{cond}.csv"
xenium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_none.csv"
visium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_none.csv"

# Function to compute mean Pearson per seed for a given file pattern
def compute_seed_means(path_template, cond_key, rep):
    means = []
    for seed in seeds:
        path = path_template.format(seed=seed, rep=rep, cond=cond_key)
        df = pd.read_csv(path, index_col=0)
        means.append(df['Pearson'].mean())
    return means

# Collect results
results = {rep: {'means': [], 'sems': []} for rep in replicates}

for rep in replicates:
    all_means = []
    all_sems = []
    # Xenium baseline
    xm_means = compute_seed_means(xenium_path, None, rep)
    all_means.append(np.mean(xm_means))
    all_sems.append(np.std(xm_means, ddof=1) / np.sqrt(len(xm_means)))
    # Sparsity thresholds
    for cond in [5, 15, 45]:
        sp_means = compute_seed_means(sparsity_path, cond, rep)
        all_means.append(np.mean(sp_means))
        all_sems.append(np.std(sp_means, ddof=1) / np.sqrt(len(sp_means)))
    # Visium baseline
    vis_means = compute_seed_means(visium_path, None, rep)
    all_means.append(np.mean(vis_means))
    all_sems.append(np.std(vis_means, ddof=1) / np.sqrt(len(vis_means)))
    
    results[rep]['means'] = all_means
    results[rep]['sems'] = all_sems


conditions = ['Xenium'] + [f'L{cond}' for cond in [ 5, 15, 45]] + ['Visium']

x = np.arange(len(conditions))
fig, ax = plt.subplots(figsize=(8, 5))

# Plot 'test' in color C0, 'rep2' in color C1
for idx, rep in enumerate(replicates):
    y = results[rep]['means']
    yerr = results[rep]['sems']
    color = f'C{idx}'  # C0 for test, C1 for rep2
    # Connect all except final Visium point
    ax.errorbar(x[:-1], y[:-1], yerr=yerr[:-1], fmt='--o', color=color, label=f'{rep}', barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8)
    # Plot final Visium point without line
    ax.errorbar(x[-1], y[-1], yerr=yerr[-1], fmt='o', color=color, barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Mean Pearson Correlation')
ax.set_title('Pearson vs. Condition across Replicates')
ax.legend(title='Replicate')
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("fig3d_poisson.svg", dpi=1000, bbox_inches="tight")



##############################################################################################################


## figure for visium rescue experiment ##


# # read in file names
# file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/rescue_visium/')
# # get xenium and visium file names
# # # NOTE getting rid of SAVER for now = breastcancer_visium_res14_seed42_aligned_correlation_summary_SAVER.txt
# file_names_test = [file for file in file_names if 'test' in file and 'summary' in file]

# # read in file names
# table_visium_test = make_table(file_names_test, '/home/caleb/Desktop/improvedgenepred/results/rescue_visium/')
# table_visium_test

# # rename index
# table_visium_test.index = ['KNN', 'MAGIC', 'SCVI', 'Visium', 'Xenium']

# # reorder the index
# desired_order = ['Visium', 'KNN', 'MAGIC', 'SCVI', 'Xenium']
# table_visium_test = table_visium_test.reindex(desired_order)



# # read in file names
# file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/rescue_visium/')
# # get visium
# # NOTE getting rid of SAVER for now = breastcancer_visium_res14_seed42_aligned_correlation_summary_SAVER.txt
# file_names_rep2 = [file for file in file_names if 'rep2' in file and 'summary' in file]

# # make tables
# table_visium_rep2 = make_table(file_names_rep2, '/home/caleb/Desktop/improvedgenepred/results/rescue_visium/')
# table_visium_rep2

# # rename index
# table_visium_rep2.index = ['KNN', 'MAGIC', 'SCVI', 'Visium', 'Xenium']

# # reorder the index
# desired_order = ['Visium', 'KNN', 'MAGIC', 'SCVI', 'Xenium']
# table_visium_rep2 = table_visium_rep2.reindex(desired_order)



# # plot the index as the x-axis and the pearson correlation as the y-axis
# sns.lineplot(x=table_visium_test.index, y='pearson', data=table_visium_test, label='Test Set', marker='o', markersize=10)
# sns.lineplot(x=table_visium_rep2.index, y='pearson', data=table_visium_rep2, label='Rep2', marker='o', markersize=10)
# # plot lines connecting the x-axis and y-axis
# # for i in range(len(table_visium)):
# #     plt.plot([table_visium.index[i], table_visium_rep2.index[i]], [table_visium['pearson'][i], table_visium_rep2['pearson'][i]], color='black')
# # yticks every .1
# # plt.yticks(np.arange(-.3, 1.1, step=0.1))
# plt.grid(axis='y')
# plt.xlabel('Method')
# plt.ylabel('Pearson Correlation')
# plt.title('Visium Data Visium Image')
# plt.savefig("fig3e_rescue.svg", dpi=1000, bbox_inches="tight")
# # sns.despine()




# NEW rescue analysis #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define seeds,  thresholds, and replicates
seeds = [42, 0, 1, 10, 100]
conditions = ['Visium', 'KNN', 'MAGIC', 'SCVI','Xenium']
replicates = ['test', 'rep2']

# Base path templates
sparsity_path = "/home/caleb/Desktop/improvedgenepred/results/rescue_visium/" \
                "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_{cond}.csv"
xenium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_xeniumdata_xeniumimage_seed{seed}_{rep}_correlation_df_none.csv"
visium_path = "/home/caleb/Desktop/improvedgenepred/results/original_four/" \
               "breastcancer_visiumdata_visiumimage_seed{seed}_{rep}_correlation_df_none.csv"

# Function to compute mean Pearson per seed for a given file pattern
def compute_seed_means(path_template, cond_key, rep):
    means = []
    for seed in seeds:
        path = path_template.format(seed=seed, rep=rep, cond=cond_key)
        df = pd.read_csv(path, index_col=0)
        means.append(df['Pearson'].mean())
    return means

# Collect results
results = {rep: {'means': [], 'sems': []} for rep in replicates}

for rep in replicates:
    all_means = []
    all_sems = []
    # Visium baseline
    vis_means = compute_seed_means(visium_path, None, rep)
    all_means.append(np.mean(vis_means))
    all_sems.append(np.std(vis_means, ddof=1) / np.sqrt(len(vis_means)))
    # Sparsity thresholds
    for cond in ['KNN', 'MAGIC', 'SCVI']:
        sp_means = compute_seed_means(sparsity_path, cond, rep)
        print(cond, sp_means)
        all_means.append(np.mean(sp_means))
        all_sems.append(np.std(sp_means, ddof=1) / np.sqrt(len(sp_means)))

    # Xenium baseline
    xm_means = compute_seed_means(xenium_path, None, rep)
    all_means.append(np.mean(xm_means))
    all_sems.append(np.std(xm_means, ddof=1) / np.sqrt(len(xm_means)))
    
    results[rep]['means'] = all_means
    results[rep]['sems'] = all_sems


# conditions = ['Xenium'] + [f'{cond}' for cond in ['KNN', 'MAGIC', 'SCVI']] + ['Visium']

x = np.arange(len(conditions))
fig, ax = plt.subplots(figsize=(8, 5))

# Plot 'test' in color C0, 'rep2' in color C1
for idx, rep in enumerate(replicates):
    y = results[rep]['means']
    yerr = results[rep]['sems']
    color = f'C{idx}'  # C0 for test, C1 for rep2
    # Plot all points individually without connecting lines
    ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, label=f'{rep}', barsabove=True, capsize=8, elinewidth=0.5, markeredgewidth=0.5, linewidth=0.5, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Mean Pearson Correlation')
ax.set_title('Pearson vs. Condition across Replicates')
ax.legend(title='Replicate')
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("fig3e_rescue.svg", dpi=1000, bbox_inches="tight")



##############################################################################################################


##############################################################################################################


### sparsity analysis ###


# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/sparsity')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'df' in file and 'full' not in file and 'test' in file] 

# sort strings
file_names_xenium.sort()

# read in data
visium = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_test_correlation_df_none.csv', index_col=0)
xenium = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_test_correlation_df_none.csv', index_col=0)

lessthan0 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[0], index_col=0)
lessthan1 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[1], index_col=0)
lessthan10 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[2], index_col=0)
lessthan15 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[3], index_col=0)
lessthan20 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[4], index_col=0)
lessthan5 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[5], index_col=0)

# combine all pearson correaltion columns to a new dataframe, where column is the name of the file and rows are the pearson correlation values so i can plot a boxplot
plot_variable = "Pearson"
z = pd.DataFrame({
    'xenium': xenium[plot_variable],
    'lessthan0': lessthan0[plot_variable],
    'lessthan1': lessthan1[plot_variable],
    'lessthan5': lessthan5[plot_variable],
    'lessthan10': lessthan10[plot_variable],
    'lessthan15': lessthan15[plot_variable],
    'lessthan20': lessthan20[plot_variable],
    'visium': visium[plot_variable]
})
z

desired_order = ['xenium', '<=0', '<=1', '<=5', '<=10', '<=15', '<=20', 'visium']


# read in file names
file_names = os.listdir('/home/caleb/Desktop/improvedgenepred/results/sparsity')
# get xenium and visium file names
file_names_xenium = [file for file in file_names if 'xenium' in file and 'df' in file and 'full' not in file and 'rep2' in file] 

# sort strings
file_names_xenium.sort()

visium = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_visiumdata_visiumimage_seed42_rep2_correlation_df_none.csv', index_col=0)
xenium = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_xeniumdata_xeniumimage_seed42_rep2_correlation_df_none.csv', index_col=0)

lessthan0 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[0], index_col=0)
lessthan1 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[1], index_col=0)
lessthan10 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[2], index_col=0)
lessthan15 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[3], index_col=0)
lessthan20 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[4], index_col=0)
lessthan5 = pd.read_csv('/home/caleb/Desktop/improvedgenepred/results/sparsity/' + file_names_xenium[5], index_col=0)

# combine all pearson correaltion columns to a new dataframe, where column is the name of the file and rows are the pearson correlation values so i can plot a boxplot
z_rep2 = pd.DataFrame({
    'xenium': xenium[plot_variable],
    'lessthan0': lessthan0[plot_variable],
    'lessthan1': lessthan1[plot_variable],
    'lessthan5': lessthan5[plot_variable],
    'lessthan10': lessthan10[plot_variable],
    'lessthan15': lessthan15[plot_variable],
    'lessthan20': lessthan20[plot_variable],
    'visium': visium[plot_variable]
})
z_rep2





# Convert to long format
z_long = z.melt(var_name='Method', value_name='Pearson Correlation')
z_long['Replicate'] = 'Test Set'

z_rep2_long = z_rep2.melt(var_name='Method', value_name='Pearson Correlation')
z_rep2_long['Replicate'] = 'Rep2'

# Combine the dataframes
df_long = pd.concat([z_long, z_rep2_long])

# reset index
df_long = df_long.reset_index(drop=True)






# plot a boxplot of the total number of genes used
plt.figure(figsize=(10, 1))

# count total non na in each column
z_count = z.count()
z_rep2_count = z_rep2.count()

# plot a bar plot of the total number of genes used side by side
plt.bar(range(len(z_count)), z_count, width=0.4, label='Test Set', color='#8FA9C8')
# plt.bar(np.arange(len(z_rep2_count)) + 0.4, z_rep2_count, width=0.4, label='Rep2', color='#DF9B59')
# set x ticks
plt.xticks(np.arange(len(z_count)), desired_order)
plt.ylabel('Total Genes Used')
sns.despine()
# plot grid
plt.grid(axis='y' ,  linewidth=0.5)
plt.yticks([0, 100, 200, 306])
# plt.savefig("fig3_sparsitybar.svg", dpi=1000, bbox_inches="tight")



##############################################################################################################

### read in aligned data ###

import scanpy as sc
import scipy.sparse

# # combined data
# adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_data.h5ad')
# adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/visium_data_poissonlambda5.h5ad')
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_poisson_noise/xenium_data_poissonlambda45.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)


# Calculate variance of each gene (column) for adata_xenium
gene_variances_xen = np.array(adata_xenium.X.toarray().var(axis=0)).flatten()
gene_variances_vis = np.array(adata_visium.X.toarray().var(axis=0)).flatten()


# Plot histogram of gene variances
plt.figure(figsize=(8, 4))
plt.hist(gene_variances_vis, bins=50, color="#55B4E9", edgecolor="black", alpha=0.5, label='Visium')
plt.hist(gene_variances_xen, bins=50, color="#E69F01", edgecolor="black", alpha=0.5, label='Xenium')
plt.xlabel("Gene Variance")
plt.ylabel("Frequency")
plt.title("Histogram of Gene Variances (Xenium)")
plt.legend()
plt.tight_layout()
plt.show()


adata_visium.X.toarray()
adata_xenium.X.toarray()