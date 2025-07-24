'''
Misc functions for the pipeline
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from anndata import AnnData
import time
from scipy.sparse import csr_matrix


####################################################################################################


# Function to subset AnnData by bounding box coordinates
def subset_by_patch(adata, x_start, x_end, y_start, y_end):
    """Subset an AnnData object based on a spatial range."""
    # Extract the spatial coordinates (assumed to be under obsm["spatial"])
    spatial_coords = adata.obsm["spatial"]

    # Create a mask to filter spots within the bounding box using NumPy's logical_and
    mask = np.logical_and.reduce((
        spatial_coords[:, 0] >= x_start,
        spatial_coords[:, 0] <= x_end,
        spatial_coords[:, 1] >= y_start,
        spatial_coords[:, 1] <= y_end
    ))

    # Subset the AnnData object based on the mask
    return adata[mask, :].copy()


####################################################################################################


# Extract patches using `extract_patches_from_centers` function

def extract_and_subset_patches(image, adata, centers, patch_size=100):
    """Extract patches from an image and subset AnnData accordingly."""
    adata_sub_dict = {}
    half_size = patch_size // 2
    img_height, img_width, _ = image.shape

    centers = np.array(centers)

    x_starts = np.maximum(centers[:, 0] - half_size, 0)
    x_ends = np.minimum(centers[:, 0] + half_size, img_width - 1)
    y_starts = np.maximum(centers[:, 1] - half_size, 0)
    y_ends = np.minimum(centers[:, 1] + half_size, img_height - 1)

    start_time = time.time()
    total_centers = len(centers)

    for i, (x_start, x_end, y_start, y_end) in enumerate(zip(x_starts, x_ends, y_starts, y_ends)):
        # Extract the patch from the image (if needed, else skip this step)
        patch = image[y_start:y_end, x_start:x_end, :]

        # Subset the AnnData object based on the bounding box
        adata_patch = subset_by_patch(adata, x_start, x_end, y_start, y_end)

        # Store the patch data in the AnnData object
        adata_patch.uns['spatial'] = patch

        # Create a unique name for the patch
        patch_name = f"patch_{i}"
        adata_sub_dict[patch_name] = adata_patch

        # Calculate elapsed time and estimated remaining time
        elapsed_time = time.time() - start_time
        avg_time_per_iteration = elapsed_time / (i + 1)
        remaining_time = avg_time_per_iteration * (total_centers - (i + 1))

        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        remaining_minutes = int(remaining_time // 60)
        remaining_seconds = int(remaining_time % 60)

        if i % 500 == 0:
            # Print progress
            print(f"Processed {i + 1}/{total_centers} patches. "
                  f"Elapsed time: {elapsed_minutes}m {elapsed_seconds}s. "
                  f"Estimated remaining time: {remaining_minutes}m {remaining_seconds}s.")

    return adata_sub_dict

####################################################################################################


def plot_xenium_with_centers(adata, gene_list, g, patch_centers, patch_size=100, if_pred=True):
    """
    Plots the expression of a specified gene over an image using patch centers.

    :param adata: The AnnData object containing expression data and the image in `uns['spatial']`.
    :param gene_list: List of genes available in the expression data.
    :param g: The specific gene to visualize.
    :param patch_centers: Array of shape (N, 2) indicating the centers of each patch.
    :param patch_size: Size of the square patch (length of one side).
    :param if_pred: Boolean indicating whether to use "Predicted" or "True" expression title.
    """
    # Ensure the image is available in adata's uns
    if 'spatial' not in adata.uns:
        raise ValueError("The image data is missing in `adata.uns['spatial']`.")

    # Ensure the specified gene is available
    if g not in gene_list:
        raise ValueError(f"Gene '{g}' not found in the provided gene list.")

    # Extract the image from adata's uns
    image = adata.uns['spatial']

    # Create a matplotlib figure with the image in the background
    fig, ax = plt.subplots()
    ax.imshow(image)

    # make array
    adata.X_array = pd.DataFrame(adata.X.toarray(), index=adata.obs.index)

    # Normalize the expression data of the specified gene
    gene_idx = gene_list.index(g)
    values = np.array(adata.X_array.iloc[:, gene_idx])
    norm = plt.Normalize(values.min(), values.max())

    # Create a colormap for the gene expression data
    scalar_map = plt.cm.ScalarMappable(norm=norm)

    # Calculate the top-left corner for each patch
    half_patch_size = patch_size // 2
    top_left_corners = patch_centers - np.array([half_patch_size, half_patch_size])

    # Plot each rectangular patch with the gene expression value, skipping missing data
    for i, (top_left_x, top_left_y) in enumerate(top_left_corners):
        if np.isnan(values[i]):
            continue  # Skip patches without valid gene expression values

        # Map the expression value to a color
        square_color = scalar_map.to_rgba(values[i])

        # Create and add the rectangle patch
        ax.add_patch(patches.Rectangle(
            (top_left_x, top_left_y), patch_size, patch_size,
            linewidth=1, edgecolor='none', facecolor=square_color, alpha=1
        ))

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the color bar
    scalar_map.set_array([])
    fig.colorbar(scalar_map, ax=ax, orientation='vertical')

    # Determine and set the plot title
    title_prefix = "Predicted" if if_pred else "True"
    ax.set_title(f"{title_prefix} Expression of {g}")

    # Display the plot
    plt.show()



import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as T
import pytorch_lightning as pl

# Custom dataset class to include the external list of indices
class CustomTensorDataset(Dataset):
    """Dataset wrapping tensors or arrays and spatial coordinates."""
    def __init__(self, indices, coords, image_array, y_array, mode='train'):
        """
        :param indices: List of indices (e.g., external references)
        :param coords: Spatial coordinates for each data point
        :param image_array: NumPy array with shape (N, H, W, C)
        :param y_array: NumPy array with shape (N, num_labels)
        :param mode: 'train' or 'val' to determine whether to apply transformations
        """
        assert image_array.shape[0] == y_array.shape[0], "Image and label arrays must have the same first dimension"
        self.indices = indices
        self.coords = coords
        self.mode = mode

        # Convert NumPy arrays to PyTorch tensors
        self.image_tensors = torch.tensor(image_array, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to [N, C, H, W]
        self.y_tensors = torch.tensor(y_array, dtype=torch.float32)

        # Define image transformations for training
        self.transform = T.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.2255))
        ])

        # Define image transformations for validation (only normalization)
        self.no_transform = T.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        img_tensor = self.image_tensors[index]
        target_tensor = self.y_tensors[index]

        # Apply transformations to image tensor based on mode
        if self.mode == 'train':
            img_tensor = self.transform(img_tensor)
        else:
            img_tensor = self.no_transform(img_tensor)

        # Return the data, target, the corresponding external index, and coordinates
        return (img_tensor, target_tensor, self.indices[index], self.coords[index])

    def __len__(self):
        return self.image_tensors.size(0)

class GeneExpressionDataModule(pl.LightningDataModule):
    def __init__(self, indices, coords, X_train, y_train,batch_size=32, mode='train'):
        super().__init__()
        self.indices = indices
        self.coords = coords
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.mode = mode

    def train_dataloader(self):
        train_dataset = CustomTensorDataset(self.indices, self.coords, self.X_train, self.y_train, mode=self.mode)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_dataloader



# import 
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim import Adam
import random
from torch.utils.data import random_split


# Set the random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Worker seed function for DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomTensorDatasetValidation(Dataset):
    """
    Dataset wrapping tensors or arrays and spatial coordinates.

    Args:
        indices (list): List of indices (e.g., external references).
        coords (array-like): Spatial coordinates for each data point.
        image_array (np.ndarray): Image data with shape (N, H, W, C).
        y_array (np.ndarray): Target labels with shape (N, num_labels).
        mode (str): Either 'train', 'val', or 'test' to determine the transformation pipeline.
    """
    def __init__(self, indices, coords, image_array, y_array, mode='train'):
        assert image_array.shape[0] == y_array.shape[0], "Image and label arrays must have the same first dimension"
        self.indices = indices
        self.coords = coords
        self.mode = mode

        # Normalize image array to [0, 1] if in uint8 format
        if image_array.dtype == np.uint8:
            image_array = image_array / 255.0

        # Convert images and labels to PyTorch tensors
        self.image_tensors = torch.tensor(image_array, dtype=torch.float32).permute(0, 3, 1, 2)  # [N, C, H, W]
        self.y_tensors = torch.tensor(y_array, dtype=torch.float32)

        # Define default transformations
        self.transforms = {
            'train': T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(90),
                T.Resize((224, 224)),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                
            ]),
            'val': T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.Resize((224, 224)),

            ]),
            'test': T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.Resize((224, 224)),
            ]),
        }

    def __getitem__(self, index):
        img_tensor = self.image_tensors[index]
        target_tensor = self.y_tensors[index]

        # Apply transformations based on mode
        img_tensor = self.transforms[self.mode](img_tensor)

        # Return the data, target, index, and spatial coordinates
        return img_tensor, target_tensor, self.indices[index], self.coords[index]

    def __len__(self):
        return self.image_tensors.size(0)




class GeneExpressionDataModuleValidation(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for gene expression data with train, validation, and test splits.

    Args:
        indices (list): List of indices for the dataset.
        coords (array-like): Spatial coordinates for each data point.
        X_data (np.ndarray): Image data with shape (N, H, W, C).
        y_data (np.ndarray): Target labels with shape (N, num_labels).
        batch_size (int): Batch size for dataloaders.
        val_pct (float): Proportion of data to use for validation.
        test_pct (float): Proportion of data to use for testing.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, indices, coords, X_data, y_data, batch_size=32, val_pct=0.2, test_pct=0.1, seed=42):
        super().__init__()
        self.indices = indices
        self.coords = coords
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.seed = seed

    def setup(self, stage=None):
        # Set the seed for reproducibility
        set_seed(self.seed)

        # Calculate the number of samples for each dataset
        total_size = len(self.X_data)
        val_size = int(total_size * self.val_pct)
        test_size = int(total_size * self.test_pct)
        train_size = total_size - val_size - test_size

        # Create the datasets with appropriate modes
        dataset = CustomTensorDatasetValidation(self.indices, self.coords, self.X_data, self.y_data, mode='train')
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Assign modes
        train_dataset.dataset.mode = 'train'
        val_dataset.dataset.mode = 'val'
        test_dataset.dataset.mode = 'test'

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker)

# Define the model class
class GeneExpressionPredictor(pl.LightningModule):
    def __init__(self, output_size, dropout_rate=0.1, method="visium", lossplot_save_file = "/home/caleb/Desktop/improvedgenepred/results/loss_plots/loss_plot.png"):
        super().__init__()
        self.epoch_losses = []
        self.val_losses = []
        self.method = method
        self.lossplot_save_file = lossplot_save_file

        # Feature extractor (ResNet)
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer

        hidden_sizes = [2048, 1024, 512, 256]
        # Feature processing layers with increased complexity
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[
                layer for size in zip(hidden_sizes[:-1], hidden_sizes[1:])
                for layer in (nn.Linear(size[0], size[1]), nn.BatchNorm1d(size[1]), nn.ReLU(), nn.Dropout(dropout_rate))
            ],
        )

        self.output = nn.Linear(256, output_size)

    def forward(self, patches):
        x = self.feature_extractor(patches)
        x = self.feature_layers(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        test_loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss_epoch"]
        avg_val_loss = self.trainer.callback_metrics["val_loss_epoch"]
        self.epoch_losses.append(avg_loss.item())
        self.val_losses.append(avg_val_loss.item())

    def on_train_end(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Training and Validation Loss Per Epoch')
        plt.legend()
        plt.savefig(self.lossplot_save_file)
        # plt.show()

    def predict(self, patches):
        self.eval()
        with torch.no_grad():
            predictions = self(patches)
        return predictions




# Define the model class
class GeneExpressionPredictorTransformer(pl.LightningModule):
    def __init__(self, output_size, dropout_rate=0.1, method="visium", lossplot_save_file = "/home/caleb/Desktop/improvedgenepred/results/loss_plots/loss_plot.png"):
        super().__init__()
        self.epoch_losses = []
        self.val_losses = []
        self.method = method
        self.lossplot_save_file = lossplot_save_file

        # Feature extractor (ResNet)
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer

        # Transformer-based feature processing
        self.embedding_dim = 2048  # Match the ResNet50 output features
        self.feature_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=dropout_rate,
                activation='relu'
            ),
            num_layers=4
        )


        hidden_sizes = [2048, 1024, 512, 256, output_size]
        # Feature processing layers with increased complexity
        self.output = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[
                layer for size in zip(hidden_sizes[:-1], hidden_sizes[1:])
                for layer in (nn.Linear(size[0], size[1]), nn.BatchNorm1d(size[1]), nn.ReLU(), nn.Dropout(dropout_rate))
            ],
        )
        # self.output = nn.Linear(256, output_size)

    def forward(self, patches):
        x = self.feature_extractor(patches)
        x = self.feature_layers(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        test_loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss_epoch"]
        avg_val_loss = self.trainer.callback_metrics["val_loss_epoch"]
        self.epoch_losses.append(avg_loss.item())
        self.val_losses.append(avg_val_loss.item())

    def on_train_end(self):
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Training and Validation Loss Per Epoch')
        plt.legend()
        plt.savefig(self.lossplot_save_file)
        plt.show()

    def predict(self, patches):
        self.eval()
        with torch.no_grad():
            predictions = self(patches)
        return predictions
    


'''
SEraster functions for the pipeline
Author: Caleb
Date: 2024-07-24
'''

# import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from anndata import AnnData
import time
import pandas as pd
import scanpy as sc
from PIL import Image
import anndata as ad
from scipy.sparse import csr_matrix
import matplotlib.colorbar as mcolorbar
import matplotlib.patches as mpatches




# Function to subset and aggregate AnnData by bounding box coordinates
def subset_and_aggregate_patch(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation='mean',visium=False):
    """Subset an AnnData object based on a spatial range and aggregate the data, ensuring cells are only included in the first patch they appear in."""
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    # filter spots within the bounding box and not already used
    mask = (
        (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) &
        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
    )
    
    # Remove cells that have already been used
    mask = mask & (~adata.obs.index.isin(used_cells))

    # Subset the AnnData object based on the mask
    adata_patch = adata[mask, :]

    # Return None if there are no cells in the patch
    if adata_patch.shape[0] == 0:
        return None

    # Add these cells to the set of used cells
    used_cells.update(adata_patch.obs.index)

    # Aggregate the data within the patch
    if aggregation == 'sum':
        aggregated_data = adata_patch.X.sum(axis=0)
    elif aggregation == 'mean':
        aggregated_data = adata_patch.X.mean(axis=0)
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")

    # Create a new AnnData object with aggregated data
    aggregated_data = aggregated_data if isinstance(aggregated_data, csr_matrix) else csr_matrix(aggregated_data)
    new_adata = ad.AnnData(X=aggregated_data)
    
    # Add image patch
    new_adata.uns['spatial'] = image[y_start:y_end, x_start:x_end]
    # Add patch coordinates
    new_adata.uns['patch_coords'] = [x_start, x_end, y_start, y_end]
    
    # Add centroid of new patch
    new_adata.obs['x_centroid'] = (x_start + x_end) / 2
    new_adata.obs['y_centroid'] = (y_start + y_end) / 2

    # Aggregate and sum specific fields
    # for field in ['transcript_counts', 'control_probe_counts', 'control_codeword_counts', 'total_counts', 'cell_area', 'nucleus_area']:
    #     new_adata.obs[field] = adata_patch.obs[field].sum()

    if visium:
        for field in ['in_tissue', 'array_row', 'array_col']:
            new_adata.obs[field] = adata_patch.obs[field].iloc[0]

    # Add spatial coordinates
    new_adata.obsm["spatial"] = new_adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)

    # Add variables and gene names
    new_adata.var = adata.var
    new_adata.var_names = adata.var_names

    return new_adata

# Function to extract patches and aggregate data from an image and AnnData object
def rasterizeGeneExpression_topatches(image, adata, patch_size=100, aggregation='mean', visium=False):
    """Extract non-overlapping patches from an image and aggregate AnnData data accordingly."""

    # Initialize variables
    adata_sub_dict = {}
    img_height, img_width, _ = image.shape
    used_cells = set()

    if visium:
        # Determine the bounding box using the centroids - same method as SEraster
        x_min = (np.floor(adata.obsm['spatial'][:,0].min()) - patch_size/2).astype(int)
        x_max = (np.ceil(adata.obsm['spatial'][:,0].max()) + patch_size/2).astype(int)
        y_min = (np.floor(adata.obsm['spatial'][:,1].min()) - patch_size/2).astype(int)
        y_max = (np.ceil(adata.obsm['spatial'][:,1].max()) + patch_size/2).astype(int)
    else:
        # Determine the bounding box using the centroids - same method as SEraster
        x_min = (np.floor(adata.obs['x_centroid'].min()) - patch_size/2).astype(int)
        x_max = (np.ceil(adata.obs['x_centroid'].max()) + patch_size/2).astype(int)
        y_min = (np.floor(adata.obs['y_centroid'].min()) - patch_size/2).astype(int)
        y_max = (np.ceil(adata.obs['y_centroid'].max()) + patch_size/2).astype(int)

    # init start points
    x_starts = np.arange(x_min, x_max, patch_size)
    y_starts = np.arange(y_min, y_max, patch_size)

    # get total patches
    total_patches = len(x_starts) * len(y_starts)
    patch_index = 0

    # Loop through all patches
    for y_start in y_starts:
        for x_start in x_starts:
            x_end = min(x_start + patch_size, img_width)
            y_end = min(y_start + patch_size, img_height)

            # Subset and aggregate the AnnData object
            adata_patch = subset_and_aggregate_patch(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation, visium)
            
            # Filter out empty patches
            if adata_patch is not None:
                if adata_patch.uns['spatial'].shape == (patch_size, patch_size, 3):
                    patch_name = f"patch_{patch_index}"
                    adata_sub_dict[patch_name] = adata_patch

            patch_index += 1

    # return the dictionary of patches
    return adata_sub_dict



# Function to plot patches on the original image
def plotRaster(image, adata_patches, color_by='gene_expression', gene_name=None):
    """
    Plots patches on the original image, colored by either gene expression or a column in adata_patches.obs.

    Parameters:
    - image: The original image array.
    - adata_patches: Dictionary of AnnData objects representing the patches.
    - color_by: How to color the patches ('gene_expression' or 'total_expression').
    - gene_name: The name of the gene to use if color_by is 'gene_expression'.
    """
    # Check inputs
    if color_by == 'gene_expression' and gene_name is None:
        raise ValueError("You must specify a gene_name when color_by='gene_expression'.")

    # Collect all values for normalization
    values = []
    for adata_patch in adata_patches.values():
        if color_by == 'gene_expression':
            expression = adata_patch.X[:, adata_patch.var_names.get_loc(gene_name)].sum()
            values.append(expression)
        elif color_by == 'total_expression':
            total_expression = adata_patch.X.sum()
            values.append(total_expression)
    
    # get min and max values
    values = np.array(values)
    min_value, max_value = values.min(), values.max()

    # Plot the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # Plot each patch with the appropriate color
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
        
        # Draw a rectangle for the patch
        rect = mpatches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                  linewidth=1, edgecolor='none', facecolor=color, alpha=1)
        ax.add_patch(rect)

    # Create a color bar
    norm = plt.Normalize(min_value, max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(f'{gene_name} Expression' if color_by == 'gene_expression' else "total_expression")

    plt.axis('off')
    plt.show()






import numpy as np
from skimage.metrics import structural_similarity as ssim

def make_patch_raster(image_shape, adata_patches, gene_name):
    """
    Build a raster image from patches, coloring each patch by its gene expression,
    and leave background as NaN.
    """
    H, W = image_shape[:2]
    # initialize everything to NaN
    raster = np.full((H, W), 9999999, dtype=float)

    # collect expressions for normalization
    exprs = []
    for ad in adata_patches.values():
        exprs.append(ad.X[:, ad.var_names.get_loc(gene_name)].sum())
    exprs = np.array(exprs, float)
    lo, hi = exprs.min(), exprs.max()
    span = hi - lo if hi > lo else 1.0

    # fill each patch region
    for ad, val in zip(adata_patches.values(), exprs):
        x0, x1, y0, y1 = ad.uns['patch_coords']
        norm = (val - lo) / span
        raster[y0:y1, x0:x1] = norm

    return raster



def compute_raster_ssim(
    img_shape,
    adata_patches_true,
    adata_patches_pred,
    gene_name='xxx',
    min_patch_area=4,
    **ssim_kwargs
):
    """
    Compute 2D SSIM between predicted and true patch rasters,
    treating background (NaNs) as excluded.
    """

    # build NaN‐initialized rasters
    R_pred = make_patch_raster(img_shape, adata_patches_pred, gene_name)
    R_true = make_patch_raster(img_shape, adata_patches_true, gene_name)

    # mask = valid wherever BOTH rasters are not NaN (or just either, depending on what you want)
    mask = (R_true != 9999999) | (R_pred != 9999999)

    # require enough pixels for at least one window
    if mask.sum() < min_patch_area:
        return np.nan

    # compute SSIM with mask support (scikit-image ≥0.16)
    s = ssim(
        R_true[mask],
        R_pred[mask],
        data_range=1.0,
        # mask=mask,
        multichannel=False,
        **ssim_kwargs
    )
    return s


import numpy as np

def calculate_ssim(x, y, num_breaks=256):
    """
    Compute a global 1D SSIM-like index between two vectors x and y.
    
    Steps:
      1. Flatten & normalize each to [0,1] by its own max.
      2. Discretize into `num_breaks` levels (0 … num_breaks-1).
      3. Compute means, variances, covariance (unbiased, ddof=1).
      4. Plug into the SSIM formula.
    
    Returns a float between -1 and 1.
    """
    # --- 1) flatten & to float ---
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # --- 2) normalize to [0,1] (or all-zero if max==0) ---
    mx = x.max()
    x = x/mx if mx > 0 else np.zeros_like(x)
    my = y.max()
    y = y/my if my > 0 else np.zeros_like(y)

    # --- 3) discretize into num_breaks bins 0..(num_breaks-1) ---
    # Equivalent to R's cut(..., labels=FALSE)-1
    # We map [0,1] → [0, num_breaks-1] by floor(x*(num_breaks-1))
    levels = num_breaks - 1
    xdig = np.floor(x * levels).astype(int)
    ydig = np.floor(y * levels).astype(int)

    # --- 4) SSIM constants ---
    C1 = (0.01 * levels) ** 2
    C2 = (0.03 * levels) ** 2

    # --- 5) means, variances (ddof=1), covariance (ddof=1) ---
    mux = xdig.mean()
    muy = ydig.mean()
    sigx = xdig.var(ddof=1)
    sigy = ydig.var(ddof=1)
    sigxy = np.cov(xdig, ydig, ddof=1)[0, 1]

    # --- 6) SSIM formula ---
    num = (2 * mux * muy + C1) * (2 * sigxy + C2)
    den = (mux**2 + muy**2 + C1) * (sigx + sigy + C2)
    ssim = num / den

    # clamp just in case of tiny numerical overshoot
    return float(np.clip(ssim, -1.0, 1.0))




import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_regression

def evaluate_model_validation(data_module, model, combined_adata, output_file="evaluation_output.txt", save_per_patch=False):
    """
    Evaluates the model on the test set and calculates correlations, MSE, rMSE,
    SSIM, mutual information, and Jensen–Shannon divergence.
    """
    # --- (unchanged setup & prediction code) ---
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    model.eval()

    all_predictions, all_indices = [], []
    with torch.no_grad():
        for batch in test_loader:
            data, target, indices, coords = batch
            data = data.to(model.device)
            preds = model(data)
            all_predictions.append(preds.cpu())
            all_indices.extend(indices)

    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    # Prepare combined_adata_copy and results DataFrame (unchanged)
    combined_adata_copy = combined_adata.copy()
    combined_adata_copy = combined_adata_copy[combined_adata_copy.obs.index.isin(all_indices)]
    combined_adata_copy.X_array = pd.DataFrame(
        combined_adata_copy.X.toarray(),
        index=combined_adata_copy.obs.index
    )
    combined_adata_copy.X_array.index = combined_adata_copy.X_array.index.str.replace("patch_", "").astype(int)

    results = pd.DataFrame(all_predictions, index=all_indices)
    results.index = results.index.str.replace("patch_", "").astype(int)
    results.sort_index(inplace=True)

    adata_pred = combined_adata_copy.copy()
    adata_pred = adata_pred[adata_pred.obs.index.isin(all_indices)]
    adata_pred.X_array = results
    adata_pred.X = csr_matrix(adata_pred.X_array)

    # Compute overall MSE
    total_mse = np.mean((adata_pred.X_array - combined_adata_copy.X_array)**2)

    # Initialize lists for per-gene metrics
    corrs_spearman = []
    corrs_pearson  = []
    per_gene_mse   = []
    per_gene_rmse  = []
    per_gene_norm_rmse_range = []
    per_gene_ssim  = []
    per_gene_mi    = []
    per_gene_jsd   = []

    # Per-gene loop
    for i in range(results.shape[1]):
        # print every 50
        if i % 50 == 0:
            print(f"Processing gene {i+1}/{results.shape[1]}: {combined_adata_copy.var_names[i]}")
        pred = results.iloc[:, i].values
        actual = combined_adata_copy.X_array.iloc[:, i].values

        # Correlations
        corrs_spearman.append(stats.spearmanr(pred, actual)[0])
        corrs_pearson.append(stats.pearsonr(pred, actual)[0])

        # MSE & rMSE
        mse_i  = np.mean((pred - actual)**2)
        rmse_i = np.sqrt(mse_i)
        per_gene_mse.append(mse_i)
        per_gene_rmse.append(rmse_i)

        # Normalized rMSE
        rng = actual.max() - actual.min()
        per_gene_norm_rmse_range.append(rmse_i / rng if rng else 0)

    
        # SSIM 
        try:
            ssim_val = calculate_ssim(pred, actual, num_breaks=256)
        except ValueError:
            ssim_val = np.nan
        per_gene_ssim.append(ssim_val)

        # Mutual Information
        try:
            mi_val = mutual_info_regression(
                pred.reshape(-1,1),
                actual,
                random_state=0
            )[0]
        except Exception:
            mi_val = np.nan
        per_gene_mi.append(mi_val)

        # Jensen–Shannon Divergence
        # clip
        pred = np.clip(pred, 0, None)  # avoid log(0)
        actual = np.clip(actual, 0, None)  # avoid log(0)
        # add small value to avoid division by zero
        pred = np.maximum(pred, 1e-12)
        actual = np.maximum(actual, 1e-12)
        # normalize to sum to 1
        p = actual / actual.sum()
        q = pred   / pred.sum()
        # compute JSD
        jsd = jensenshannon(p, q)**2  # scipy returns sqrt(JS); square to get divergence
        per_gene_jsd.append(jsd)

    # Write summary stats to file
    with open(output_file, "w") as f:
        f.write(f"Mean Spearman: {np.nanmean(corrs_spearman)}\n")
        f.write(f"Mean Pearson:  {np.nanmean(corrs_pearson)}\n")
        f.write(f"Mean MSE:     {np.nanmean(per_gene_mse)}\n")
        f.write(f"Total MSE:      {total_mse}\n")
        f.write(f"Mean rMSE:    {np.nanmean(per_gene_rmse)}\n")
        f.write(f"Mean rMSE range: {np.nanmean(per_gene_norm_rmse_range)}\n")
        f.write(f"Mean SSIM:    {np.nanmean(per_gene_ssim)}\n")
        f.write(f"Mean Mutual Info: {np.nanmean(per_gene_mi)}\n")
        f.write(f"Mean JSD:     {np.nanmean(per_gene_jsd)}\n")

    # Build the final DataFrame
    correlation_df = pd.DataFrame({
        "Gene": combined_adata.var.index,
        "Spearman": corrs_spearman,
        "Pearson":  corrs_pearson,
        "MSE":      per_gene_mse,
        "rMSE":     per_gene_rmse,
        "rMSE_range": per_gene_norm_rmse_range,
        "SSIM":       per_gene_ssim,
        "MutualInfo": per_gene_mi,
        "JSDivergence": per_gene_jsd
    }).sort_values(by="Pearson", ascending=False)

    return correlation_df, adata_pred









def save_per_patch_results(data_module, model, combined_adata, output_file="evaluation_output.csv"):
    """
    Evaluates the model on the test set and calculates correlations, MSE, rMSE,
    SSIM, mutual information, and Jensen–Shannon divergence.
    """
    # --- (unchanged setup & prediction code) ---
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    model.eval()

    all_predictions, all_indices = [], []
    with torch.no_grad():
        for batch in test_loader:
            data, target, indices, coords = batch
            data = data.to(model.device)
            preds = model(data)
            all_predictions.append(preds.cpu())
            all_indices.extend(indices)

    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    # Prepare combined_adata_copy and results DataFrame (unchanged)
    combined_adata_copy = combined_adata.copy()
    combined_adata_copy = combined_adata_copy[combined_adata_copy.obs.index.isin(all_indices)]
    combined_adata_copy.X_array = pd.DataFrame(
        combined_adata_copy.X.toarray(),
        index=combined_adata_copy.obs.index
    )
    combined_adata_copy.X_array.index = combined_adata_copy.X_array.index.str.replace("patch_", "").astype(int)

    results = pd.DataFrame(all_predictions, index=all_indices)
    results.index = results.index.str.replace("patch_", "").astype(int)
    results.sort_index(inplace=True)

    results.columns = combined_adata_copy.var_names

    # save
    results.to_csv(output_file)



import torch
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def evaluate_and_save_results(model, data_module, combined_adata, file_name, resolution, save_results=True, output_path=None):
    """
    Evaluates the model on the provided data and computes metrics such as MSE and correlation. Optionally saves the results.

    Parameters:
    - model (torch.nn.Module): The trained model to evaluate.
    - data_module (pl.LightningDataModule): The data module providing the data loaders.
    - combined_adata (AnnData): The AnnData object containing the ground truth data.
    - file_name (str): The base name for the output files.
    - resolution (int): The resolution parameter for the output file naming.
    - save_results (bool): Whether to save the results to disk. Default is True.
    - output_path (str): The directory where to save the results. Required if save_results is True.

    Returns:
    - total_mse (float): The mean squared error between the predictions and actual values.
    - correlation_df (pd.DataFrame): A DataFrame containing gene correlations.
    """
    # Set the device and move the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Containers for predictions and indices
    all_predictions = []
    all_indices = []

    with torch.no_grad():
        for batch in data_module.train_dataloader():
            data, target, indices, coords = batch
            data = data.to(device)

            # Get predictions
            predictions = model(data)

            # Store predictions and indices
            all_predictions.append(predictions.cpu())
            all_indices.append(indices)

    # Concatenate predictions and indices
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_indices = torch.cat(all_indices, dim=0).numpy()

    # Prepare the AnnData object
    adata_seraster1 = combined_adata.copy()
    adata_seraster1.X = adata_seraster1.X.toarray()

    # Create a DataFrame with predictions
    resultsss = pd.DataFrame(all_predictions, index=all_indices)
    resultsss = resultsss.groupby(resultsss.index).mean()

    # Prepare the predictions AnnData object
    adata_pred = adata_seraster1.copy()
    adata_pred.X_array = resultsss

    # Compute the total MSE
    total_mse = np.mean((adata_pred.X - adata_seraster1.X)**2)

    # Compute correlations
    corrs_spearman = []
    corrs_pearson = []
    for i in range(resultsss.shape[1]):
        corrs_spearman.append(stats.spearmanr(resultsss.iloc[:, i], adata_seraster1.X[:, i])[0])
        corrs_pearson.append(stats.pearsonr(resultsss.iloc[:, i], adata_seraster1.X[:, i])[0])

    # Print mean correlations
    print("Spearman correlation: ", np.nanmean(corrs_spearman))
    print("Pearson correlation: ", np.nanmean(corrs_pearson))
    print("Total MSE: ", total_mse)

    # Create a DataFrame of correlations
    correlation_df = pd.DataFrame({"Gene": combined_adata.var.index, "Spearman": corrs_spearman, "Pearson": corrs_pearson})
    correlation_df = correlation_df.sort_values(by="Spearman", ascending=False)
    print(correlation_df)

    # Optionally save the results
    if save_results and output_path is not None:
        output_file = f"{output_path}/{file_name}_res{resolution}_correlation_df.csv"
        correlation_df.to_csv(output_file, index=False)

    # Plot the Pearson correlation distribution
    plt.hist(corrs_pearson, bins=20)
    plt.title("Pearson Correlation Distribution")
    plt.xlim(-1, 1)
    plt.show()

    return correlation_df, adata_pred




def combine_adata_patches(adata_patches, image):
    # Initialize list to collect data
    adata_list = []

    # Iterate over the dictionary to prepare the data for concatenation
    for key, adata in adata_patches.items():
        # Set the index for each observation to the dictionary key
        adata.obs.index = [key] * adata.shape[0]
        adata_list.append(adata)

    # Concatenate all the adata objects
    combined_adata = ad.concat(adata_list, merge='same', uns_merge='same')
    # add image
    combined_adata.uns['spatial'] = image
    # add X_array
    combined_adata.X_array = pd.DataFrame(combined_adata.X.toarray(), index=combined_adata.obs.index)

    return combined_adata




import torch
import numpy as np
import re

def prepare_data(adata_patches):
    # Prepare image tensors, spatial coordinates, and correct order
    image_tensors = [
        np.array(adata_patches[key].uns['spatial'])
        for key in adata_patches
        for _ in range(len(adata_patches[key]))
    ]
    spatial_coords = np.vstack([adata_patches[key].obsm["spatial"] for key in adata_patches])
    correct_order = [key for key in adata_patches for _ in range(len(adata_patches[key]))]

    # Scale spatial coordinates if needed
    spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32)

    # Prepare target gene expression data
    y_train = np.concatenate(
        [adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X for adata in adata_patches.values()],
        axis=0
    )

    # Stack image tensors into a single array
    X_train = np.stack(image_tensors, axis=0)

    print("Final image array shape:", X_train.shape)
    print("Shape of concatenated y_train:", y_train.shape)

    return X_train, y_train, spatial_coords, correct_order







import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from transformers import ViTModel, ViTFeatureExtractor
import torch
import matplotlib.pyplot as plt

# Define the model class
class GeneExpressionPredictor_VIT(pl.LightningModule):
    def __init__(self, output_size, dropout_rate=0.1, method="visium"):
        super().__init__()
        self.epoch_losses = []
        self.val_losses = []
        self.method = method

        # Pretrained Vision Transformer (ViT) as feature extractor
        self.feature_extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.feature_dim = self.feature_extractor.config.hidden_size

        hidden_sizes = [self.feature_dim, 1024, 512, 256]
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[
                layer for size in zip(hidden_sizes[:-1], hidden_sizes[1:])
                for layer in (nn.Linear(size[0], size[1]), nn.BatchNorm1d(size[1]), nn.ReLU(), nn.Dropout(dropout_rate))
            ],
        )

        self.output = nn.Linear(256, output_size)

    def forward(self, patches):
        # Transform patches with the transformer model and extract the last hidden state
        vit_outputs = self.feature_extractor(pixel_values=patches).last_hidden_state
        # Global average pooling on the hidden states
        x = vit_outputs.mean(dim=1)
        x = self.feature_layers(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        patches, y, index, coords = batch
        y_hat = self(patches)
        test_loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss_epoch"]
        avg_val_loss = self.trainer.callback_metrics["val_loss_epoch"]
        self.epoch_losses.append(avg_loss.item())
        self.val_losses.append(avg_val_loss.item())

    def on_train_end(self):
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Training and Validation Loss Per Epoch')
        plt.legend()
        plt.savefig('/home/caleb/Desktop/improvedgenepred/results/loss_plots/loss_plot.png')
        plt.show()

    def predict(self, patches):
        self.eval()
        with torch.no_grad():
            predictions = self(patches)
        return predictions





def transform_coordinates(x, y, image_width, image_height, rotation_k=0, flip_axis=None):
    """
    Transforms (x, y) coordinates to match image transformations using np.flip first, then np.rot90.
    
    Parameters:
    - x: Array of original x-coordinates.
    - y: Array of original y-coordinates.
    - image_width: Width of the image.
    - image_height: Height of the image.
    - rotation_k: Number of 90-degree rotations counterclockwise (0, 1, 2, or 3).
    - flip_axis: Axis to flip (None, 0 for vertical, 1 for horizontal).
    
    Returns:
    - x_new: Transformed x-coordinates (array of integers).
    - y_new: Transformed y-coordinates (array of integers).
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Step 1: Apply flipping using np.flip
    if flip_axis == 0:  # Vertical flip
        y = image_height - 1 - y
    elif flip_axis == 1:  # Horizontal flip
        x = image_width - 1 - x

    # Step 2: Apply rotation using np.rot90
    if rotation_k % 4 == 1:  # 90 degrees counterclockwise
        x_new = image_height - 1 - y
        y_new = x
    elif rotation_k % 4 == 2:  # 180 degrees
        x_new = image_width - 1 - x
        y_new = image_height - 1 - y
    elif rotation_k % 4 == 3:  # 270 degrees counterclockwise (90 degrees clockwise)
        x_new = y
        y_new = image_width - 1 - x
    else:  # rotation_k % 4 == 0, no rotation
        x_new, y_new = x, y

    # Ensure the final coordinates are integers
    x_new = np.round(x_new).astype(int)
    y_new = np.round(y_new).astype(int)

    return x_new, y_new


from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from functools import partial



# from torchvision.models import resnet50

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu_1 = nn.ReLU(inplace=False)
        self.relu_2 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_1(out) # 1 MODIFIED!!

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu_2(out) # 2 MODIFIED!!

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu_1 = nn.ReLU(inplace=False)
        self.relu_2 = nn.ReLU(inplace=False)
        self.relu_3 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_1(out) # 1 MODIFIED!!

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_2(out) # 2 MODIFIED!!

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu_3(out) # 3 MODIFIED!!

        return out



class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # 1
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2

# @register_model()
@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

# resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
# resnet_model.fc = nn.Identity()  # Remove the final classification layer


# Function to subset and aggregate AnnData by bounding box coordinates
def subset_and_aggregate_patch_basedoncenters(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation='mean', visium=False):
    """Subset an AnnData object based on a spatial range and aggregate the data, ensuring cells are only included in the first patch they appear in."""
    # Extract spatial coordinates
    spatial_coords = adata.obsm["spatial"]

    # filter spots within the bounding box and not already used
    mask = (
        (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) &
        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
    )
    
    # Remove cells that have already been used
    mask = mask & (~adata.obs.index.isin(used_cells))

    # Subset the AnnData object based on the mask
    adata_patch = adata[mask, :]

    # Return None if there are no cells in the patch
    if adata_patch.shape[0] == 0:
        return None

    # Add these cells to the set of used cells
    used_cells.update(adata_patch.obs.index)

    # Aggregate the data within the patch
    if aggregation == 'sum':
        aggregated_data = adata_patch.X.sum(axis=0)
    elif aggregation == 'mean':
        aggregated_data = adata_patch.X.mean(axis=0)
    else:
        raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'.")

    # Create a new AnnData object with aggregated data
    aggregated_data = aggregated_data if isinstance(aggregated_data, csr_matrix) else csr_matrix(aggregated_data)
    new_adata = ad.AnnData(X=aggregated_data)
    
    # Add image patch
    new_adata.uns['spatial'] = image[y_start:y_end, x_start:x_end]
    # Add patch coordinates
    new_adata.uns['patch_coords'] = [x_start, x_end, y_start, y_end]
    
    # Add centroid of new patch
    new_adata.obs['x_centroid'] = (x_start + x_end) / 2
    new_adata.obs['y_centroid'] = (y_start + y_end) / 2

    if visium:
        for field in ['in_tissue', 'array_row', 'array_col']:
            new_adata.obs[field] = adata_patch.obs[field].iloc[0]

    # Add spatial coordinates
    new_adata.obsm["spatial"] = new_adata.obs[["x_centroid", "y_centroid"]].to_numpy().astype(int)

    # Add variables and gene names
    new_adata.var = adata.var
    new_adata.var_names = adata.var_names
    # make sure X is a sparse matrix
    new_adata.X = csr_matrix(new_adata.X)

    return new_adata

# Function to extract patches and aggregate data from an image and AnnData object based on supplied center coordinates
def rasterizeGeneExpression_topatches_basedoncenters(image, adata, center_coords, patch_size=100, aggregation='mean', visium=False):
    """Extract patches centered around supplied coordinates from an image and aggregate AnnData data accordingly."""

    # Initialize variables
    adata_sub_dict = {}
    img_height, img_width, _ = image.shape
    used_cells = set()

    # Loop through each center coordinate
    for patch_index, (x_center, y_center) in enumerate(center_coords):
        # Calculate bounding box around the center coordinate
        x_start = max(0, x_center - patch_size // 2)
        x_end = min(img_width, x_center + patch_size // 2)
        y_start = max(0, y_center - patch_size // 2)
        y_end = min(img_height, y_center + patch_size // 2)

        # Subset and aggregate the AnnData object
        adata_patch = subset_and_aggregate_patch_basedoncenters(adata, image, x_start, x_end, y_start, y_end, used_cells, aggregation, visium)
        
        # Filter out empty patches
        if adata_patch is not None:
            if adata_patch.uns['spatial'].shape == (patch_size, patch_size, 3):
                patch_name = f"patch_{patch_index}"
                adata_sub_dict[patch_name] = adata_patch

    # return the dictionary of patches
    return adata_sub_dict

