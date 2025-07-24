
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

# # read in svg results
# gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
# gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# # these were not in the data
# gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]


# ### read in aligned data ###

# # # combined data
# adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_data.h5ad')
# adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_data.h5ad')
# # adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_data.h5ad')
# # adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_data.h5ad')

# # make .X a csr matrix
# adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
# adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# # add array for gene expression
# adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
# adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# # # patches
# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/visiumdata_xeniumimage_patches.pkl', 'rb') as f:
#     aligned_visium_dictionary = pickle.load(f)

# with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/xeniumdata_xeniumimage_patches.pkl', 'rb') as f:
#     aligned_xenium_dictionary = pickle.load(f)

# # # patches
# # with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/visiumdata_visiumimage_patches.pkl', 'rb') as f:
# #     aligned_visium_dictionary = pickle.load(f)

# # with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_tovisiumimage/xeniumdata_visiumimage_patches.pkl', 'rb') as f:
# #     aligned_xenium_dictionary = pickle.load(f)

# # plt.imshow(adata_visium.uns['spatial'])

# # scale the data
# scaling_factor = 1
# for i in aligned_visium_dictionary:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
#     # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))


# # scale the data
# scaling_factor = 1
# for i in aligned_xenium_dictionary:
#     # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
#     aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
#     # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# # log transform the data
# sc.pp.log1p(adata_xenium)
# sc.pp.log1p(adata_visium)

# # choose method
# # method = "visium"
# method = "xenium"

# # img
# # image_version = "visium"
# image_version = "xenium"

# # resolution based on image
# resolution = 250

# # edits
# edit = "none"

# # prepare the datasets
# if method == "visium":
#     X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary)
# else:
#     X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary)

# gb = 0

########################################################################################################


### read in for gb ###

# read in svg results
gene_list = pd.read_csv("/home/caleb/Desktop/improvedgenepred/data/breastcancer_xenium_sample1_rep1/rastGexp_df.csv", index_col=0)
gene_list = [gene for gene in gene_list.index if "BLANK" not in gene and "Neg" not in gene and  "antisense" not in gene]
# these were not in the data
gene_list = [gene for gene in gene_list if gene not in ['AKR1C1', 'ANGPT2', 'APOBEC3B', 'BTNL9', 'CD8B', 'POLR2J3', 'TPSAB1']]



gb = 125
# combined data
adata_xenium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_xenium_raw_gb' + str(gb) + '.h5ad')
adata_visium = sc.read_h5ad('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/combined_aligned_visium_raw_gb' + str(gb) + '.h5ad')

# make .X a csr matrix
adata_xenium.X = scipy.sparse.csr_matrix(adata_xenium.X)
adata_visium.X = scipy.sparse.csr_matrix(adata_visium.X)

# add array for gene expression
adata_xenium.X_array = pd.DataFrame(adata_xenium.X.toarray(), index=adata_xenium.obs.index)
adata_visium.X_array = pd.DataFrame(adata_visium.X.toarray(), index=adata_visium.obs.index)

# patches
with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_visium_dictionary_raw_gb' + str(gb) + '.pkl', 'rb') as f:
    aligned_visium_dictionary = pickle.load(f)

with open('/home/caleb/Desktop/improvedgenepred/data/breastcancer_sample1_rep1_aligned_toxeniumimage/aligned_xenium_dictionary_raw_gb' + str(gb) + '.pkl', 'rb') as f:
    aligned_xenium_dictionary = pickle.load(f)

# plt.imshow(adata_visium.uns['spatial'])

# scale the data
scaling_factor = 1
for i in aligned_visium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_visium_dictionary[i].X = sc.pp.log1p(np.round(aligned_visium_dictionary[i].X * scaling_factor))
    # aligned_visium_dictionary[i].X = sc.pp.scale(np.round(aligned_visium_dictionary[i].X * scaling_factor))

# adata_visium.X_array = adata_visium.X_array * scaling_factor

# scale the data
scaling_factor = 1
for i in aligned_xenium_dictionary:
    # aligned_visium_dictionary[i].X_array = sc.pp.log1p(aligned_visium_dictionary[i].X_array * scaling_factor)
    aligned_xenium_dictionary[i].X = sc.pp.log1p(np.round(aligned_xenium_dictionary[i].X * scaling_factor))
    # aligned_xenium_dictionary[i].X = sc.pp.scale(np.round(aligned_xenium_dictionary[i].X * scaling_factor))


# log transform the data
sc.pp.log1p(adata_xenium)
sc.pp.log1p(adata_visium)

# choose method
# method = "visium"
method = "xenium"
image_version = "xenium"

# resolution based on image
resolution = 250

# edits
edit = 'gb' + str(gb)

# prepare the datasets
if method == "visium":
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_visium_dictionary)
else:
    X_train, y_train, scaled_coords, correct_order = prepare_data(aligned_xenium_dictionary)




#############################################################################################################


# set seed
seed_val = 42
# Define the model
set_seed(seed_val)  # Set the seed for reproducibility



data_module = GeneExpressionDataModuleValidation(
    indices=correct_order,
    coords=scaled_coords,
    X_data=X_train,
    y_data=y_train,
    batch_size=64,
    val_pct=0.10,
    test_pct=0.15,
    seed=seed_val
)


# # Prepare datasets and dataloaders
# data_module.setup()

# # Access dataloaders
# train_loader = data_module.train_dataloader()
# val_loader = data_module.val_dataloader()
# test_loader = data_module.test_dataloader()


output_size = adata_visium.shape[1]  # Assuming adata is defined
epochs = 150
model = GeneExpressionPredictor(output_size, dropout_rate=0.2, method = method, lossplot_save_file = f'/home/caleb/Desktop/improvedgenepred/results/original_four/breastcancer_' + method + 'data_' + image_version + 'image_seed' + str(seed_val) + '_' + edit + '_lossplot.png')
print(model)

# # Load the model
# model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/results/original_four/model_{method}data_{image_version}image_epochs{epochs}_seed{seed_val}_{edit}.pth"))
model.load_state_dict(torch.load(f"/home/caleb/Desktop/improvedgenepred/results/img_quality/model_{method}data_{image_version}image_epochs{epochs}_seed{seed_val}_{edit}.pth"))


# correlation_df, adata_pred = evaluate_model_validation(
#     data_module,
#     model,
#     adata_xenium,
#     output_file=f'/home/caleb/Desktop/improvedgenepred/results/delete.txt'
# )




#############################################################################################################

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_module.setup("test")
batch = next(iter(data_module.test_dataloader()))
data, _, _, _ = batch
data = data.to(device)

# Use a small background set (e.g. first 16 patches)
background = data[:32]
test_images = data[32:42]  # five test patches


### gradcam ###
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_idx: int,
    target_layer: torch.nn.Module
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for one image and one target output index.

    Args:
      model:         a PyTorch model that, given (1, C, H, W), returns (1, num_outputs).
      image:         a single image tensor of shape (C, H, W), already normalized.
      target_idx:    the index of the output neuron (e.g. gene/class) you care about.
      target_layer:  the convolutional layer (nn.Module) you want to hook.

    Returns:
      A numpy array heatmap of shape (H, W), values in [0,1].
    """

    device = next(model.parameters()).device
    model.eval()

    # These will store, respectively, the activation maps and the gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        # output has shape (1, C, H', W')
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        # grad_output[0] also has shape (1, C, H', W')
        gradients.append(grad_output[0].detach())

    # 1) Register exactly ONE forward hook
    handle_fw = target_layer.register_forward_hook(forward_hook)

    # 2) Register exactly ONE backward hook (choose full_backward if available)
    # if hasattr(target_layer, "register_full_backward_hook"):
    #     handle_bw = target_layer.register_full_backward_hook(backward_hook)
    # else:
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # 3) Prepare a single‐image batch (1, C, H, W)
    input_tensor = image.unsqueeze(0).to(device).requires_grad_(True)

    # 4) Forward pass
    model.zero_grad()
    outputs = model(input_tensor)                 # shape: (1, num_outputs)
    score   = outputs[0, target_idx]              # scalar
    score.backward()                              # triggers backward_hook

    # 5) Grab the stored activation and gradient
    act = activations.pop()   # shape: (1, C, H', W')
    grad = gradients.pop()    # shape: (1, C, H', W')

    # 6) Remove hooks so they don’t accumulate next time
    handle_fw.remove()
    handle_bw.remove()

    # 7) Compute channel‐wise weights: α_k = mean over spatial dimensions of ∂y/∂A_k
    # grad.shape = (1, C, H', W')
    # weights = grad.mean(dim=(2, 3), keepdim=True)    # → (1, C, 1, 1)
    ### GRAD-CAM+ ###
    # option A: clamp negatives to zero
    weights = grad.clamp(min=0).mean(dim=(2, 3), keepdim=True)

    # 8) Weighted sum of activation maps: ∑ₖ αₖ·Aₖ
    cam_map = (weights * act).sum(dim=1, keepdim=True)  # → (1, 1, H', W')
    cam_map = F.relu(cam_map)                           # ReLU

    # 9) Convert to numpy and normalize to [0,1]
    cam = cam_map[0, 0].cpu().numpy()  # shape: (H', W')
    if cam.max() != cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # 10) Upsample to original image size
    _, _, H, W = input_tensor.shape  # (1, C, H, W)
    cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0).to(device)
    cam_resized = F.interpolate(cam_tensor, size=(H, W), mode="bilinear", align_corners=False)
    heatmap = cam_resized[0, 0].cpu().numpy()  # shape: (H, W)

    return heatmap

def visualize_gradcam(
    image_denorm: np.ndarray,
    heatmap: np.ndarray,
    cmap="jet",
    alpha=0.5
):
    """
    Overlay the Grad-CAM heatmap on a denormalized RGB image.

    Args:
      image_denorm: (H, W, 3) float array in [0,1].
      heatmap:      (H, W) float array in [0,1].
      cmap:         colormap for the heatmap.
      alpha:        transparency for overlay.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(image_denorm)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis("off")
    # plt.show()


# =======================
# USAGE EXAMPLE (adapt to your variables)
# =======================

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = "ADIPOQ" # PDGFRA, CD4, CD19, ADIPOQ

# gb = 0

# Plot Grad-CAM for the first 5 test images
# for i in range(len(test_images)):
for i in range(3):
    test_image = test_images[i].to(device)
    img_np = test_image.detach().cpu().numpy().transpose(1, 2, 0)
    img_denorm = np.clip(img_np * std + mean, 0.0, 1.0)
    target_layer = model.feature_extractor.layer4[-1]
    target_gene_idx = gene_list.index(g)  # g should be defined, e.g. g = "TOMM7"
    heatmap = gradcam(model.to(device), test_image, target_gene_idx, target_layer)
    visualize_gradcam(img_denorm, heatmap, cmap="jet", alpha=0.5)
    plt.savefig(f"/home/caleb/Desktop/improvedgenepred/05_figures/figure4/gradcam_{method}_{gb}_{g}_test{i}.png", bbox_inches='tight', dpi=300)
    # save image
    plt.imshow(img_denorm)
    plt.axis("off")
    plt.savefig(f"/home/caleb/Desktop/improvedgenepred/05_figures/figure4/image_{method}_{gb}_{g}_test{i}.png", bbox_inches='tight', dpi=300)


