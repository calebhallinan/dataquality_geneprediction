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


# Custom dataset class to include the external list of indices
class CustomTensorDataset_validation(Dataset):
    """Dataset wrapping tensors or arrays and spatial coordinates."""
    def __init__(self, indices, coords, image_array, y_array, mode='train'):
        """
        :param indices: List of indices (e.g., external references)
        :param coords: Spatial coordinates for each data point
        :param image_array: NumPy array with shape (N, H, W, C)
        :param y_array: NumPy array with shape (N, num_labels)
        :param mode: 'train' or 'val' or 'test' to determine whether to apply transformations
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
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # added
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # added
            # transforms.RandomGrayscale(p=0.1), # added
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Define image transformations for validation and testing (only normalization)
        self.no_transform = T.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        img_tensor = self.image_tensors[index]
        target_tensor = self.y_tensors[index]

        # Apply transformations to image tensor based on mode
        if self.mode == 'train':
            img_tensor = self.transform(img_tensor)
                # Debug: Save an example transformed image to check
            # if index < 5:  # Save the first few images
            #     transforms.ToPILImage()(img_tensor).save(f"/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/data/augmented_image_{index}.png")
        else:
            img_tensor = self.no_transform(img_tensor)

        # Return the data, target, the corresponding external index, and coordinates
        return (img_tensor, target_tensor, self.indices[index], self.coords[index])

    def __len__(self):
        return self.image_tensors.size(0)

class GeneExpressionDataModule_validation(pl.LightningDataModule):
    def __init__(self, indices, coords, X_data, y_data, batch_size=32, val_pct=0.2, test_pct=0.1, mode='train', seed=42):
        super().__init__()
        self.indices = indices
        self.coords = coords
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.mode = mode
        self.seed = seed

    def setup(self, stage=None):
        # Set the seed for reproducibility
        set_seed(self.seed)

        # Calculate the number of samples for each dataset
        total_size = len(self.X_data)
        val_size = int(total_size * self.val_pct)
        test_size = int(total_size * self.test_pct)
        train_size = total_size - val_size - test_size

        # Create the datasets
        dataset = CustomTensorDataset_validation(self.indices, self.coords, self.X_data, self.y_data, mode=self.mode)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.seed))

        # Update the modes for validation and testing datasets
        self.val_dataset.dataset.mode = 'val'
        self.test_dataset.dataset.mode = 'test'
        self.train_dataset.dataset.mode = 'train'

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker)

# Define the model class
class GeneExpressionPredictor(pl.LightningModule):
    def __init__(self, output_size, dropout_rate=0.1, method="visium"):
        super().__init__()
        self.epoch_losses = []
        self.val_losses = []
        self.method = method

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
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.title('Training and Validation Loss Per Epoch')
        plt.legend()
        plt.savefig('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/pipeline/results_final/loss_plot.png')
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



import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def evaluate_model_validation(data_module, model, combined_adata):
    """
    Evaluates the model on the test set and calculates correlations and MSE.

    Parameters:
    - data_module: PyTorch Lightning data module with test data setup.
    - model: The trained PyTorch model.
    - combined_adata: An AnnData object containing the actual gene expression data.

    Returns:
    - correlation_df: DataFrame containing Spearman and Pearson correlations for each gene.
    - adata_pred: AnnData object containing the predicted gene expression data.
    """
    # Set up test data
    data_module.setup(stage='test')
    model.eval()

    all_predictions = []
    all_indices = []

    # Make predictions
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            data, target, indices, coords = batch
            data = data.to(model.device)
            predictions = model(data)
            all_predictions.append(predictions.cpu())
            all_indices.append(indices)

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_indices = torch.cat(all_indices, dim=0).numpy()
    all_indices = ["patch_" + str(index) for index in all_indices]

    # Filter and prepare data
    combined_adata_copy = combined_adata.copy()
    combined_adata_copy = combined_adata_copy[combined_adata_copy.obs.index.isin(all_indices)]
    combined_adata_copy.X_array = pd.DataFrame(combined_adata_copy.X.toarray(), index=combined_adata_copy.obs.index)
    combined_adata_copy.X_array.index = combined_adata_copy.X_array.index.str.replace("patch_", "").astype(int)

    # Aggregate predictions
    results = pd.DataFrame(all_predictions, index=all_indices)
    # results = results.groupby(results.index).mean()
    results.index = results.index.str.replace("patch_", "").astype(int)
    adata_pred = combined_adata_copy.copy()
    adata_pred = adata_pred[adata_pred.obs.index.isin(all_indices)]
    # order results
    results = results.loc[adata_pred.obs.index.str.replace("patch_", "").astype(int)]
    adata_pred.X = results
    adata_pred.X_array = results
    # results.sort_index(inplace=True)

    # Compute MSE
    total_mse = np.mean((adata_pred.X_array - combined_adata_copy.X_array)**2)

    # Compute correlations
    corrs_spearman = []
    corrs_pearson = []
    for i in range(results.shape[1]):
        corrs_spearman.append(stats.spearmanr(results.iloc[:, i], combined_adata_copy.X_array.iloc[:, i])[0])
        corrs_pearson.append(stats.pearsonr(results.iloc[:, i], combined_adata_copy.X_array.iloc[:, i])[0])

    # Print statistics
    print("Spearman correlation: ", np.nanmean(corrs_spearman))
    print("Pearson correlation: ", np.nanmean(corrs_pearson))
    print("Total MSE: ", total_mse)

    # Prepare correlation DataFrame
    correlation_df = pd.DataFrame({"Gene": combined_adata.var.index, "Spearman": corrs_spearman, "Pearson": corrs_pearson})
    correlation_df = correlation_df.sort_values(by="Pearson", ascending=False)
    print(correlation_df)

    # Plot Pearson correlation distribution
    plt.hist(corrs_pearson, bins=20)
    plt.title("Pearson Correlation Distribution")
    plt.xlim(-1, 1)
    plt.show()

    return correlation_df, adata_pred


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
    """
    Prepare the image tensors and spatial coordinates from the provided AnnData patches.

    Parameters:
    adata_patches (dict): Dictionary containing AnnData objects with image data under 'spatial' key and spatial coordinates under 'obsm["spatial"]'.

    Returns:
    tuple: A tuple containing:
        - X_train_tensor (torch.Tensor): The tensor containing the image data.
        - y_train_tensor (torch.Tensor): The tensor containing the gene expression data.
        - scaled_coords (torch.Tensor): The tensor containing the scaled spatial coordinates.
        - correct_order (list): List containing the correct order of samples.
    """

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize list for features and spatial coordinates
    spatial_coords = []
    correct_order = []
    image_tensors = []

    # Iterate over each key in adata_patches to extract data
    for key in adata_patches:
        img_tensor_original = adata_patches[key].uns['spatial'].copy()
        spatial_coords.append(adata_patches[key].obsm["spatial"])  # Collect spatial coordinates

        # Ensure img_tensor is an array (or convert to a numpy array if it's a different type)
        if not isinstance(img_tensor_original, np.ndarray):
            img_tensor_original = np.array(img_tensor_original)

        # Repeat img_tensor for each observation in adata_patches[key]
        num_repeats = len(adata_patches[key])

        # Append repeated image tensors to the list
        for _ in range(num_repeats):
            image_tensors.append(img_tensor_original)
            correct_order.append(key)

    # Flatten the list of lists for spatial coordinates and convert to a single tensor
    spatial_coords = np.vstack([x for xs in spatial_coords for x in xs])
    spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32)
    max_coord_value = spatial_coords.max().item()  # Find the max to scale appropriately
    scaled_coords = (spatial_coords / max_coord_value * 999).long()  # Scale and convert to integers
    correct_order = [int(re.findall(r'\d+', x)[-1]) for x in correct_order]

    # Stack all image tensors into one large numpy array along a new axis
    X_train = np.stack(image_tensors, axis=0)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

    # Collect all .X arrays from the dictionary, convert to numpy arrays, concatenate, and transfer to the device
    all_X_arrays = [adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X for adata in adata_patches.values()]
    y_train = np.concatenate(all_X_arrays, axis=0)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    print("Final image array shape:", X_train.shape)
    print("Shape of concatenated y_train:", y_train.shape)  # Debugging output

    return X_train_tensor, y_train_tensor, scaled_coords, correct_order










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
        plt.savefig('/home/caleb/Desktop/projects_caleb/histology_to_gene_prediction/pipeline/results_final/loss_plot.png')
        plt.show()

    def predict(self, patches):
        self.eval()
        with torch.no_grad():
            predictions = self(patches)
        return predictions



# function to split the data
def split_adata_patches(combined_adata, adata_patches_for_image):
    # Dictionary to store the split AnnData objects
    adata_patches = {}
    
    # Get the unique keys (patch identifiers) from the obs index
    unique_keys = combined_adata.obs.index.unique()
    
    # Iterate over the unique keys to split the data
    for key in unique_keys:
        # Subset the combined AnnData for each key
        adata_patch = combined_adata[combined_adata.obs.index == key].copy()
        # Reset the obs index to default to avoid confusion
        adata_patch.obs.index = adata_patch.obs.index.str.split('-').str[-1]

        adata_patch.uns['spatial'] = adata_patches_for_image[key].uns['spatial']
        adata_patch.uns['patch_coords'] = adata_patches_for_image[key].uns['patch_coords']
        # Store the individual AnnData object in the dictionary
        adata_patches[key] = adata_patch
    
    return adata_patches


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

