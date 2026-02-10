
# read packages
import scanpy as sc
import pandas as pd
import scipy
from utils import *
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# from UNI import *
import pickle
import datetime



##############################################################################################################################################

# method
method = 'visiumhd' # cosmx xenium visiumhd
patch_size = 210 # 250 or 210


# combined data
adata_tmp = sc.read_h5ad(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_data_{patch_size}.h5ad')


# make .X a csr matrix
adata_tmp.X = scipy.sparse.csr_matrix(adata_tmp.X)


# # patches
with open(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_patches_{patch_size}.pkl', 'rb') as f:
    patches_tmp = pickle.load(f)


# plt.imshow(adata_tmp.uns['spatial'])
# plt.axis('off')  # Remove axes
# plt.savefig(f'/home/caleb/Desktop/improvedgenepred/07_revision/figures/{method}_image.png')  # Save the image


# plotRaster(adata_tmp.uns["spatial"], patches_tmp, color_by='gene_expression', gene_name='PDYN')


####



# make data for DL
X_train, y_train, scaled_coords, correct_order = prepare_data(patches_tmp)
# quick check
X_train.shape, y_train.shape, scaled_coords.shape, len(correct_order)


print(f"Working on {method}...")

# set seed
# seed_val = 42

for seed_val in [42, 0, 1, 10, 100]:

    # Start a timer to measure execution time
    start_time = datetime.datetime.now()

    # Define the model
    set_seed(seed_val)  # Set the seed for reproducibility


    # Define the data module
    data_module = GeneExpressionDataModuleValidation(
        indices=correct_order,
        coords=scaled_coords,
        X_data=X_train,
        y_data=y_train,
        batch_size=64,
        val_pct=0.10,
        test_pct=0.15,
        seed=seed_val)

    # output size
    output_size = adata_tmp.shape[1]  # Assuming adata is defined
    epochs = 50
    model = GeneExpressionPredictor(output_size, dropout_rate=0.2, method = method, lossplot_save_file = f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_lossplot.png')
    # print(model)

    # # Trainer initialization
    trainer = pl.Trainer(max_epochs=epochs)

    # # Train the model
    trainer.fit(model, data_module)


    # # Save the model based on resolution and file name
    torch.save(model.state_dict(), f"/home/caleb/Desktop/improvedgenepred/07_revision/results/model_{method}_epochs{epochs}_seed{seed_val}.pth")


    # Load the model
    # model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/07_revision/results/model_{method}_imagetype{img_type}_epochs{epochs}_seed{seed_val}.pth"))

    # visiumhd_data_module = GeneExpressionDataModuleValidation(
    #     indices=correct_order,
    #     coords=scaled_coords,
    #     X_data=X_train,
    #     y_data=y_train,
    #     batch_size=64,
    #     val_pct=0.10,
    #     test_pct=0.15,
    #     seed=seed_val)

    # get results
    correlation_df, adata_pred = evaluate_model_validation(
        data_module,
        model,
        adata_tmp,
        output_file=f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_test_correlation_summary.txt')

    # save correlation_df
    correlation_df.to_csv(f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_test_correlation_df.csv', index=False)

    del data_module
    del correlation_df
    del adata_pred

    # # define full data module
    # data_module_full = GeneExpressionDataModuleValidation(
    #     indices=correct_order,
    #     coords=scaled_coords,
    #     X_data=X_train,
    #     y_data=y_train,
    #     batch_size=64,
    #     val_pct=0,
    #     test_pct=1,
    #     seed=seed_val)

    # # get results
    # correlation_df_full, adata_pred_full = evaluate_model_validation(
    #     data_module_full,
    #     model,
    #     adata_tmp,
    #     output_file=f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_full_correlation_summary.txt')


    # del data_module_full
    # del correlation_df_full
    # del adata_pred_full

    # Print end time
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution time for seed {seed_val}: {int(hours)}h {int(minutes)}m {int(seconds)}s")





### prediction ###


# set seed
seed_val = 42


# make data for DL
X_train, y_train, scaled_coords, correct_order = prepare_data(patches_tmp)
# quick check
X_train.shape, y_train.shape, scaled_coords.shape, len(correct_order)



# Define the model
set_seed(seed_val)  # Set the seed for reproducibility


# Define the data module
data_module = GeneExpressionDataModuleValidation(
    indices=correct_order,
    coords=scaled_coords,
    X_data=X_train,
    y_data=y_train,
    batch_size=64,
    val_pct=0.10,
    test_pct=0.15,
    seed=seed_val)

# output size
output_size = adata_tmp.shape[1]  # Assuming adata is defined
epochs = 50
model = GeneExpressionPredictor(output_size, dropout_rate=0.2, method = method, lossplot_save_file = f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_lossplot.png')
# print(model)


# Load the model
model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/07_revision/results/model_{method}_epochs{epochs}_seed{seed_val}.pth"))


# define full data module
data_module_full = GeneExpressionDataModuleValidation(
    indices=correct_order,
    coords=scaled_coords,
    X_data=X_train,
    y_data=y_train,
    batch_size=64,
    val_pct=0,
    test_pct=1,
    seed=seed_val)

# get results
correlation_df_full, adata_pred_full = evaluate_model_validation(
    data_module_full,
    model,
    adata_tmp,
    output_file=f'/home/caleb/Desktop/improvedgenepred/07_revision/results/' + method + '_patchsize' + str(patch_size) + '_seed' + str(seed_val) + '_full_correlation_summary.txt')



### plot ###

# Extract patches 
# patches_tmp_adata_pred = rasterizeGeneExpression_topatches(adata_pred.uns['spatial'], adata_pred, patch_size=patch_size, aggregation='sum', visium=True)
# len(patches_tmp_adata_pred)
# plot
# plotRaster(adata_pred.uns['spatial'], patches_tmp_adata_pred, color_by='total_expression')


# # Extract patches 
# patches_tmp_adata_pred_full = rasterizeGeneExpression_topatches(adata_pred_full.uns['spatial'], adata_pred_full, patch_size=patch_size, aggregation='sum', visium=True)
# len(patches_tmp_adata_pred_full)


# g = 'Aqp2'
# # g = 'Cyp7b1'
# # g = 'Lrp2'
# correlation_df_full[correlation_df_full['Gene'] == g]


# def plotRaster_direct(image, adata, patch_size, color_by="gene_expression", gene_name=None):
#     if color_by == "gene_expression" and gene_name is None:
#         raise ValueError("gene_name required")

#     h, w = image.shape[:2]
#     overlay = np.zeros((h, w), dtype=float)
#     counts = np.zeros((h, w), dtype=int)

#     spatial = adata.obsm["spatial"].astype(int)

#     if color_by == "gene_expression":
#         gene_idx = adata.var_names.get_loc(gene_name)
#         values = adata.X[:, gene_idx].toarray().ravel()
#     else:
#         values = np.asarray(adata.X.sum(axis=1)).ravel()

#     for (x, y), v in zip(spatial, values):
#         x0 = (x // patch_size) * patch_size
#         y0 = (y // patch_size) * patch_size
#         x1 = x0 + patch_size
#         y1 = y0 + patch_size

#         overlay[y0:y1, x0:x1] += v
#         counts[y0:y1, x0:x1] += 1

#     mask = counts > 0
#     overlay[mask] /= counts[mask]

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image)
#     im = ax.imshow(np.ma.masked_where(counts == 0, overlay), cmap="viridis")

#     plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
#     ax.axis("off")
#     plt.show()



# # # plot
# plotRaster_direct(adata_tmp.uns["spatial"],adata_pred_full,250,color_by="gene_expression",gene_name='PDYN')



# plotRaster_direct(adata_tmp.uns["spatial"],adata_tmp,250,color_by="gene_expression",gene_name='FCGBP')




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def plotRasterSideBySide_direct_and_patches(
    image1,
    adata1,
    patch_size,
    image2,
    adata_patches2,
    color_by="gene_expression",
    gene_name=None,
    save_path=None,
):
    """
    Side-by-side raster plots with a shared colorbar.
    - Left: computed directly from adata1 (no adata_patches needed)
    - Right: uses precomputed adata_patches2
    """

    if color_by == "gene_expression" and gene_name is None:
        raise ValueError("gene_name required when color_by='gene_expression'")

    # --------------------------------------------------
    # Dataset 1: compute per-patch values (no plotting)
    # --------------------------------------------------
    h, w = image1.shape[:2]
    overlay1 = np.zeros((h, w), dtype=float)
    counts1 = np.zeros((h, w), dtype=int)

    spatial = adata1.obsm["spatial"].astype(int)

    if color_by == "gene_expression":
        gene_idx = adata1.var_names.get_loc(gene_name)
        values1 = adata1.X[:, gene_idx].toarray().ravel()
    else:
        values1 = np.asarray(adata1.X.sum(axis=1)).ravel()

    for (x, y), v in zip(spatial, values1):
        x0 = (x // patch_size) * patch_size
        y0 = (y // patch_size) * patch_size
        x1 = min(x0 + patch_size, w)
        y1 = min(y0 + patch_size, h)

        overlay1[y0:y1, x0:x1] += v
        counts1[y0:y1, x0:x1] += 1

    mask1 = counts1 > 0
    overlay1[mask1] /= counts1[mask1]

    values_dataset1 = overlay1[mask1]

    # --------------------------------------------------
    # Dataset 2: collect patch values
    # --------------------------------------------------
    values_dataset2 = []
    patch_values2 = []

    for adata_patch in adata_patches2.values():
        if color_by == "gene_expression":
            v = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
        else:
            v = adata_patch.X.sum()
        patch_values2.append(v)
        values_dataset2.append(v)

    values_dataset2 = np.array(values_dataset2)

    # --------------------------------------------------
    # Shared normalization
    # --------------------------------------------------
    all_values = np.concatenate([values_dataset1, values_dataset2])
    norm = mcolors.Normalize(vmin=all_values.min(), vmax=all_values.max())
    cmap = plt.cm.viridis

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ---- Left: direct raster ----
    axes[0].imshow(image1)
    im1 = axes[0].imshow(
        np.ma.masked_where(~mask1, overlay1),
        cmap=cmap,
        norm=norm,
    )
    axes[0].set_title("Dataset 1 (direct raster)")
    axes[0].axis("off")

    # ---- Right: patch-based ----
    axes[1].imshow(image2)

    for adata_patch, v in zip(adata_patches2.values(), patch_values2):
        x_start, x_end, y_start, y_end = adata_patch.uns["patch_coords"]
        color = cmap(norm(v))
        rect = mpatches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=0,
            facecolor=color,
            alpha=1,
        )
        axes[1].add_patch(rect)

    axes[1].set_title("Dataset 2 (patch-based)")
    axes[1].axis("off")

    # --------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.08,
        pad=0.05,
        shrink=0.8,
    )
    cbar.set_label(
        f"{gene_name} Expression" if color_by == "gene_expression" else "Total Expression"
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()



g = "MAP4" # NPY2R MAP4 PDYN
plotRasterSideBySide_direct_and_patches(
    adata_tmp.uns["spatial"],
    adata_pred_full,
    210,
    adata_tmp.uns["spatial"],
    patches_tmp,
    color_by="gene_expression",
    gene_name=g,
    save_path=f"/home/caleb/Desktop/improvedgenepred/07_revision/figures/{g}_{method}_sidebysideplot.svg")

