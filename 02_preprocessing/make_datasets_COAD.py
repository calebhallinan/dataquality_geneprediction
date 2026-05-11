
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


##############################################################################################################################################


### read in data ###

# # read in adata objects
# adata_visiumhd = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/COAD/visiumhd_adata_COAD.h5ad')

# # read in image
# image_visiumhd = Image.open("/home/caleb/Desktop/improvedgenepred/data/COAD/visiumhd_image_COAD.tif")
# image_visiumhd_array = np.array(image_visiumhd)


# # rescale obsm
# adata_visiumhd.obsm['spatial'] = adata_visiumhd.obsm['spatial'].astype(float)
# adata_visiumhd.obsm['spatial'][:,0] = adata_visiumhd.obsm['spatial'][:,0]/adata_visiumhd.uns['H&E resolution']
# adata_visiumhd.obsm['spatial'][:,1] = adata_visiumhd.obsm['spatial'][:,1]/adata_visiumhd.uns['H&E resolution']


# # make var unique
# adata_visiumhd.var_names_make_unique()


# # add image to adata
# adata_visiumhd.uns['spatial'] = image_visiumhd_array

# # subset to same genelist as other 2 methods
# shared_genes = list(pd.read_csv("shared_genes_visiumhd_xenium_cosmx.csv", index_col=0)['0'])

# adata_visiumhd = adata_visiumhd[:,shared_genes]



# # # plot
# plt.imshow(adata_visiumhd.uns['spatial'])
# plt.scatter(adata_visiumhd.obsm['spatial'][:, 0], adata_visiumhd.obsm['spatial'][:, 1], s=1, c='red')



### Xenium ####


### read in data ###

# # read in adata objects
# adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/COAD/xenium_adata_COAD.h5ad')

# # read in image
# image_xenium = Image.open("/home/caleb/Desktop/improvedgenepred/data/COAD/xenium_image_COAD.tif")
# image_xenium_array = np.array(image_xenium)


# # rescale obsm
# adata_xenium.obsm['spatial'] = adata_xenium.obsm['spatial'].astype(float)
# adata_xenium.obsm['spatial'][:,0] = adata_xenium.obsm['spatial'][:,0]/adata_xenium.uns['H&E resolution'][0]
# adata_xenium.obsm['spatial'][:,1] = adata_xenium.obsm['spatial'][:,1]/adata_xenium.uns['H&E resolution'][1]


# # make var unique
# adata_xenium.var_names_make_unique()


# # add image to adata
# adata_xenium.uns['spatial'] = image_xenium_array

# # subset to same genelist as other 2 methods
# shared_genes = list(pd.read_csv("shared_genes_visiumhd_xenium_cosmx.csv", index_col=0)['0'])

# adata_xenium = adata_xenium[:,shared_genes]



# # plot
# plt.imshow(adata_xenium.uns['spatial'])
# plt.scatter(adata_xenium.obsm['spatial'][:, 0], adata_xenium.obsm['spatial'][:, 1], s=.2, c='red')






# ### cosmx ####


# ### read in data ###

# # read in adata objects
# adata_cosmx = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/COAD/cosmx_adata_COAD.h5ad')

# # read in image
# image_cosmx = Image.open("/home/caleb/Desktop/improvedgenepred/data/COAD/cosmx_image_COAD.tif")
# image_cosmx_array = np.array(image_cosmx)


# # rescale obsm
# adata_cosmx.obsm['spatial'] = adata_cosmx.obsm['spatial'].astype(float)
# adata_cosmx.obsm['spatial'][:,0] = adata_cosmx.obsm['spatial'][:,0]/adata_cosmx.uns['H&E resolution'][0]
# adata_cosmx.obsm['spatial'][:,1] = adata_cosmx.obsm['spatial'][:,1]/adata_cosmx.uns['H&E resolution'][1]


# # make var unique
# adata_cosmx.var_names_make_unique()


# # add image to adata
# adata_cosmx.uns['spatial'] = image_cosmx_array

# # subset to same genelist as other 2 methods
# shared_genes = list(pd.read_csv("shared_genes_visiumhd_xenium_cosmx.csv", index_col=0)['0'])

# adata_cosmx = adata_cosmx[:,shared_genes]

# # plot
# plt.imshow(adata_cosmx.uns['spatial'])
# plt.scatter(adata_cosmx.obsm['spatial'][:, 0], adata_cosmx.obsm['spatial'][:, 1], s=.02, c='red')




### cosmx ####


### read in data ###

# read in adata objects
adata_stereoseq = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/COAD/stereoseq_adata_COAD.h5ad')

# read in image
image_stereoseq = Image.open("/home/caleb/Desktop/improvedgenepred/data/COAD/stereoseq_image_COAD.tif")
image_stereoseq_array = np.array(image_stereoseq)


# rescale obsm
adata_stereoseq.obsm['spatial'] = adata_stereoseq.obsm['spatial'].astype(float)
adata_stereoseq.obsm['spatial'][:,0] = adata_stereoseq.obsm['spatial'][:,0]/adata_stereoseq.uns['H&E resolution']
adata_stereoseq.obsm['spatial'][:,1] = adata_stereoseq.obsm['spatial'][:,1]/adata_stereoseq.uns['H&E resolution']


# make var unique
adata_stereoseq.var_names_make_unique()


# add image to adata
adata_stereoseq.uns['spatial'] = image_stereoseq_array

# subset to same genelist as other 2 methods
shared_genes = list(pd.read_csv("shared_genes_visiumhd_xenium_cosmx.csv", index_col=0)['0'])


missing_shared_genes = [gene for gene in shared_genes if gene not in adata_stereoseq.var_names]
print("shared genes missing from adata_stereoseq.var_names:", len(missing_shared_genes))
if len(missing_shared_genes) > 0:
    print("missing genes:", missing_shared_genes[:20])

shared_genes_stereoseq = [gene for gene in shared_genes if gene not in missing_shared_genes]

len(shared_genes_stereoseq)

# subset to shared genes removing the 11
adata_stereoseq = adata_stereoseq[:,shared_genes_stereoseq]



from scipy.sparse import csr_matrix
import numpy as np
import anndata as ad


def build_patch_assignment(spatial, x_min, y_min, patch_size, n_x, n_y):
    px = ((spatial[:, 0] - x_min) // patch_size).astype(np.int32)
    py = ((spatial[:, 1] - y_min) // patch_size).astype(np.int32)

    valid = (
        (px >= 0) & (px < n_x) &
        (py >= 0) & (py < n_y)
    )

    cell_idx = np.nonzero(valid)[0]
    patch_ids = py[valid] * n_x + px[valid]

    data = np.ones(len(cell_idx), dtype=np.float32)

    A = csr_matrix(
        (data, (cell_idx, patch_ids)),
        shape=(spatial.shape[0], n_x * n_y)
    )

    return A, valid


def aggregate_all_patches(
        image,
        adata,
        patch_size=100,
        aggregation="sum",
        visium=False,
        log=False
):
    spatial = adata.obsm["spatial"]
    img_height, img_width, _ = image.shape

    if visium:
        x_min = int(np.floor(spatial[:, 0].min() - patch_size / 2))
        y_min = int(np.floor(spatial[:, 1].min() - patch_size / 2))
    else:
        x_min = int(np.floor(adata.obs["x_centroid"].min() - patch_size / 2))
        y_min = int(np.floor(adata.obs["y_centroid"].min() - patch_size / 2))

    x_min = max(0, x_min)
    y_min = max(0, y_min)

    n_x = int(np.ceil((img_width - x_min) / patch_size))
    n_y = int(np.ceil((img_height - y_min) / patch_size))
    n_patches = n_x * n_y

    A, valid = build_patch_assignment(
        spatial, x_min, y_min, patch_size, n_x, n_y
    )

    # ✅ Cells per patch (column sums)
    cell_counts = np.asarray(A.sum(axis=0)).ravel().astype(int)

    # 🚀 ONE sparse matmul
    patch_X = A.T @ adata.X

    if aggregation == "mean":
        counts = cell_counts.copy()
        counts[counts == 0] = 1
        patch_X = patch_X.multiply(1.0 / counts[:, None])

    # --------------------------------------------------
    # Build AnnData objects
    # --------------------------------------------------
    adata_sub_dict = {}
    used_patch_ids = []

    for pid in range(n_patches):
        row = patch_X[pid]
        if row.nnz == 0:
            continue

        py = pid // n_x
        px = pid % n_x

        x_start = x_min + px * patch_size
        y_start = y_min + py * patch_size
        x_end = min(x_start + patch_size, img_width)
        y_end = min(y_start + patch_size, img_height)

        if (x_end - x_start) != patch_size or (y_end - y_start) != patch_size:
            continue

        new_adata = ad.AnnData(X=row)
        new_adata.var = adata.var.copy()
        new_adata.var_names = adata.var_names

        cx = (x_start + x_end) / 2
        cy = (y_start + y_end) / 2

        new_adata.obs["x_centroid"] = [cx]
        new_adata.obs["y_centroid"] = [cy]
        new_adata.obsm["spatial"] = np.array([[int(cx), int(cy)]])

        new_adata.uns["spatial"] = image[y_start:y_end, x_start:x_end]
        new_adata.uns["patch_coords"] = [x_start, x_end, y_start, y_end]

        if log:
            import scanpy as sc
            sc.pp.log1p(new_adata)

        adata_sub_dict[f"patch_{pid}"] = new_adata
        used_patch_ids.append(pid)

        if pid % 1000 == 0:
            print(f"Processed {pid}/{n_patches} patches")

    # --------------------------------------------------
    # ✅ Report average cells per patch
    # --------------------------------------------------
    if used_patch_ids:
        avg_cells = cell_counts[used_patch_ids].mean()
    else:
        avg_cells = 0.0

    print(f"Total patches created: {len(adata_sub_dict)}")
    print(f"Average number of cells per patch: {avg_cells:.2f}")

    return adata_sub_dict






# method
method = 'stereoseq'
# NOTE: check explore_data for calculating this
# patch for visiumhd = 210
# for xenium and cosmx = 250
# for stereoseq = 110
patch_size = 110

img_type = 'tiff'



########################################


# ### visium hd ###

# # so each patch will be ~55um x 55um
# # adata_visiumhd.uns['H&E resolution']*210

# # Extract patches 
# patches_visiumhd = aggregate_all_patches(adata_visiumhd.uns['spatial'], adata_visiumhd, patch_size=patch_size, aggregation='sum', log = True, visium=True)
# len(patches_visiumhd)


# # combined adata
# visiumhd_combined = combine_adata_patches(patches_visiumhd, adata_visiumhd.uns['spatial'])


# visiumhd_combined.write(f"/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_data_{patch_size}.h5ad")

# # save the patches aligned_visium
# with open(f'/home/caleb/Desktop/improvedgenepred/data/COAD/{method}_patches_{patch_size}.pkl', 'wb') as f:
#     pickle.dump(patches_visiumhd, f)


# plotRaster(visiumhd_combined.uns["spatial"], patches_visiumhd, color_by='gene_expression', gene_name='FCGBP')



########################################

# ### Xenium ###

# # so each patch will be ~55um x 55um
# # adata_xenium.uns['H&E resolution']*250

# # Extract patches 
# patches_xenium = aggregate_all_patches(adata_xenium.uns['spatial'], adata_xenium, patch_size=patch_size, aggregation='sum', log = True, visium=True)
# len(patches_xenium)


# # combined adata
# xenium_combined = combine_adata_patches(patches_xenium, adata_xenium.uns['spatial'])


# # plot to check
# g = "FCGBP"
# plotRaster(xenium_combined.uns['spatial'], patches_xenium, color_by='gene_expression', gene_name= g)



# xenium_combined.write(f"/home/caleb/Desktop/improvedgenepred/data/COAD/xenium_data_{patch_size}.h5ad")

# # save the patches aligned_visium
# with open(f'/home/caleb/Desktop/improvedgenepred/data/COAD/xenium_patches_{patch_size}.pkl', 'wb') as f:
#     pickle.dump(patches_xenium, f)


########################################

## cosmx

# so each patch will be ~55um x 55um
adata_cosmx.uns['H&E resolution']*250

# Extract patches 
patches_cosmx = aggregate_all_patches(adata_cosmx.uns['spatial'], adata_cosmx, patch_size=patch_size, aggregation='sum', log = True, visium=True)
len(patches_cosmx)


# combined adata
cosmx_combined = combine_adata_patches(patches_cosmx, adata_cosmx.uns['spatial'])

# plot to check
g = "FCGBP"
plotRaster(cosmx_combined.uns['spatial'], patches_cosmx, color_by='gene_expression', gene_name= g)



cosmx_combined.write(f"/home/caleb/Desktop/improvedgenepred/data/COAD/cosmx_data_{patch_size}.h5ad")

# save the patches aligned_visium
with open(f'/home/caleb/Desktop/improvedgenepred/data/COAD/cosmx_patches_{patch_size}.pkl', 'wb') as f:
    pickle.dump(patches_cosmx, f)




########################################

## cosmx

# so each patch will be ~55um x 55um
adata_stereoseq.uns['H&E resolution']*110

# Extract patches 
patches_stereoseq = aggregate_all_patches(adata_stereoseq.uns['spatial'], adata_stereoseq, patch_size=patch_size, aggregation='sum', log = True, visium=True)
len(patches_stereoseq)


# combined adata
stereoseq_combined = combine_adata_patches(patches_stereoseq, adata_stereoseq.uns['spatial'])

# plot to check
g = "FCGBP"
plotRaster(stereoseq_combined.uns['spatial'], patches_stereoseq, color_by='gene_expression', gene_name= g)



stereoseq_combined.write(f"/home/caleb/Desktop/improvedgenepred/data/COAD/stereoseq_data_{patch_size}.h5ad")

# save the patches aligned_visium
with open(f'/home/caleb/Desktop/improvedgenepred/data/COAD/stereoseq_patches_{patch_size}.pkl', 'wb') as f:
    pickle.dump(patches_stereoseq, f)

