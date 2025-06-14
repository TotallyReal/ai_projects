import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.core.pylabtools import figsize
from einops.layers.torch import Rearrange
import einops
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from typing import List

class Attention(nn.Module):

    r"""
    An attention layer.
    Input: num_entries vectors in the given dimension
    1. applies the attention via the key \ queries and value, where the key \ query connection is restricted by the
       connection_matrix (0\1 matrix)
    2.
    """

    def __init__(self, dimension: int, num_entries: int, head_dim: int, connection_matrix: torch.Tensor):
        super(Attention, self).__init__()
        self.dim = dimension
        self.num_entries = num_entries
        self.head_dim = head_dim
        self.connection_matrix = connection_matrix

        self.key   = nn.Linear(dimension, self.head_dim)
        self.query = nn.Linear(dimension, self.head_dim)
        self.value = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (and output) of shape (n, m, self.dim)
        #   n - number of batches,
        #   m - positions
        keys: torch.Tensor = self.key(x)                    # (n, m, self.head_dim)
        queries = self.query(x)                             # (n, m, self.head_dim)
        weights = queries @ keys.transpose(-2, -1)          # (n, m, m)

        weights /= self.head_dim**-0.5                     # (n, m, m)
        weights = F.softmax(weights, dim = -1)              # (n, m, m) , each (m,m) matrix is stochastic

        values = self.value(x)                              # (n, m, self.dim)
        x = weights @ values                                # (n, m, self.dim)

        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Transformer(nn.Module):

    def __init__(self,
                 num_entries: int, dim_entries: int,
                 head_dim: int, connection_matrix: torch.Tensor,
                 feed_forward_hidden_dim: int, dropout: float = 0.):
        super(Transformer, self).__init__()

        self.attention_step = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(dim_entries),
                Attention(
                    dimension=dim_entries,
                    num_entries=num_entries,
                    head_dim=head_dim,
                    connection_matrix=connection_matrix
                )
            )
        )

        self.feed_forward_step = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(dim_entries),

                nn.Linear(dim_entries, feed_forward_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feed_forward_hidden_dim, dim_entries),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x):
        x = self.attention_step(x)
        x = self.feed_forward_step(x)
        return x


class VisionTransformer(nn.Module):

    def __init__(self, image_size: int, image_channels: int, patch_size: int, emb_dim: int):
        super(VisionTransformer, self).__init__()

        assert image_size % patch_size == 0, f'Patch size {patch_size} needs to divide the image size {image_size}'
        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.emb_size = emb_dim
        self.head_size = 16

        # b = batch_index, c = channel_index
        self.patcher = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.num_patches = (image_size//patch_size)**2
        self.linearizer = nn.Linear(patch_size * patch_size * image_channels, emb_dim)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.transformer = Transformer(
            self.num_patches+1, self.emb_size, self.head_size, torch.ones(self.num_patches),
            feed_forward_hidden_dim=self.emb_size
        )

        self.positional_embedding = nn.Linear(self.num_patches, emb_dim, bias = False)

        self.final_layer = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )

        self.register_buffer("pos", torch.eye(self.num_patches))         # (num_patches, num_patches)

        self.initial_state = copy.deepcopy(self.state_dict())


    def forward(self, x: torch.Tensor):

        # input:                    (n, c, image_size, image_size) for n images with c channels
        #                           num_patches = (image_size/patch_size)^2
		# 1. Decompose to patches:  (n, num_patches, (patch_size, patch_size, channels))
        # 2. Linearize:             (n, num_patches, emb_size)
        # 3. Position data: add data according to position in image
        # 4. Attention: Combine data with previous tokens


        pos = torch.eye(self.num_patches)                   # (num_patches, num_patches)
        pos = self.positional_embedding(pos)                # (num_patches, emb_size)

        # input : x                                         # (n, c, image_size, image_size)
        x = self.patcher(x)                                 # (n, num_patches, (patch_size * patch_size * channels))
        x = self.linearizer(x)                              # (n, num_patches, emb_size)
        x += pos                                            # (n, num_patches, emb_size)    TODO

        num_batches, num_patches, _ = x.shape

        # Add cls token to inputs
        cls_tokens = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b = num_batches)
        x = torch.cat([cls_tokens, x], dim=1)       # (n, num_patches + 1, emb_size)

        x = self.transformer(x)                             # (n, num_patches + 1, emb_size)
        x = x[:, 0, :]                                      # (n, emb_size)

        x = self.final_layer(x)                             # (n, 1)

        return x

    def update_init_state(self):
        self.initial_state = copy.deepcopy(self.state_dict())

    def initialize_state(self):
        self.load_state_dict(self.initial_state)




class MyImageDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.images = data["images"]
        labels = data["labels"]
        self.constant  = labels[: ,0:2]
        self.position  = labels[: ,2:4]
        self.zoom = labels[: ,4:5].squeeze()
        self.zoom_values = set(self.zoom.tolist())
        self.labels = self.position

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# dataset = MyImageDataset(r'C:\Users\eofir\PycharmProjects\shaders\julia\data\dataset.pt')
# fig, axes = plt.subplots(5,5, figsize=(10,10), constrained_layout=True)
#
# subset = (dataset.zoom == 2.0)
#
# for ax, img, v in zip(axes.reshape(-1), dataset.images[subset], dataset.position[subset]):
#     ax.imshow(img.squeeze().detach().numpy(), cmap='gray')
#
#     formatted_title = f"({v[0]:.2f}, {v[1]:.2f})"
#     ax.set_title(formatted_title)
# plt.show()
