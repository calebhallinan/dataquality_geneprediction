import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from anndata import AnnData
import time
from scipy.sparse import csr_matrix


import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim import Adam
import random
from torch.utils.data import random_split


import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download


####################################################################

### DOWNLOADING THE MODEL WEIGHTS - ONLY NEED TO RUN ONCE ###

# login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# local_dir = "/home/caleb/Desktop/improvedgenepred/data/UNI/"
# os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
# hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
# timm_kwargs = {
#             'model_name': 'vit_giant_patch14_224',
#             'img_size': 224, 
#             'patch_size': 14, 
#             'depth': 24,
#             'num_heads': 24,
#             'init_values': 1e-5, 
#             'embed_dim': 1536,
#             'mlp_ratio': 2.66667*2,
#             'num_classes': 0, 
#             'no_embed_class': True,
#             'mlp_layer': timm.layers.SwiGLUPacked, 
#             'act_layer': torch.nn.SiLU, 
#             'reg_tokens': 8, 
#             'dynamic_img_size': True
#         }
# model = timm.create_model(
#     pretrained=False, **timm_kwargs
# )
# model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
# transform = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]
# )
# model.eval()


####################################################################



# make local dir
local_dir = "/home/caleb/Desktop/improvedgenepred/data/UNI/"

# get model parameters
timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
# create model
model = timm.create_model(
    pretrained=False, **timm_kwargs
)
# load in weights
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()


# Define the model class
class UNIGeneExpressionPredictor(pl.LightningModule):
    def __init__(self, output_size, dropout_rate=0.1, method="visium", lossplot_save_file = "/home/caleb/Desktop/improvedgenepred/results/loss_plots/loss_plot.png"):
        super().__init__()
        self.epoch_losses = []
        self.val_losses = []
        self.method = method
        self.lossplot_save_file = lossplot_save_file

        # Feature extractor (UNI2-h)
        self.feature_extractor = model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze the feature extractor


        hidden_sizes = [1536, 1024, 512, 256]
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

