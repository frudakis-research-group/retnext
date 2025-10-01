## 🤖 About

## 🚀 Tutorial

Before starting, the following packages must be installed:

```bash
pip install retnext
pip install pymoxel>=0.4.0
pip install aidsorb>=2.0.0
```

> [!NOTE]
> **All examples below assume the use of the pretrained model**. Therefore, the image generation and preprocessing parameters must be configured accordingly.

### 🎨 Generate the energy images

You can generate the energy images from the CLI as following:

```bash
moxel path/to/CIFs path/to/voxels_data/ --grid_size=32 --cubic_box=30
```
Alternatively, for more fine-grained control over the materials to be processed:
```python
from moxel.utils import voxels_from_files

cifs = ['foo.cif', 'bar.cif', ...]
voxels_from_files(cifs, 'path/to/voxels_data/', grid_size=32, cubic_box=30)
```

### ❄️ Use RetNeXt as feature extractor

Energy images are passed through the pretrained model to extract 128-D features, which are then stored as an $N\times128$ matrix in a `.csv` file.

> [!TIP]
> You can use these features alone or combine them with others features (e.g. structural) to train classical machine learning algorithms (e.g. Random Forest or XGBoost).

```python
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map
from torchvision.transforms.v2 import Compose
from retnext.modules import RetNeXt
from retnext.transforms import AddChannelDim, BoltzmannFactor
from aidsorb.data import PCDDataset as VoxelsDataset


# Required for collating unlabeled samples
def collate_none(batch, *, collate_fn_map):
    return None


# Get the names of the materials
names = [f.removesuffix('.npy') for f in os.listdir('path/to/voxels_data/')]

# Preprocessing transformations
transform_x = Compose([AddChannelDim(), BoltzmannFactor()])

# Create the dataset
dataset = VoxelsDataset(names, path_to_X='path_to_voxels_data, transform_x=transform_x)

# Create the dataloader (adjust batch_size and num_workers)
dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=256, num_workers=8)
default_collate_fn_map.update({NoneType: collate_none})

# Load pretrained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RetNeXt(pretrained=True).to(device)

# Freeze the model
model.requires_grad_(False)
model.eval()
model.fc = torch.nn.Identity()  # So .forward() returns the embeddings.

# Extract features
Z = torch.cat([model(x.to(device)) for x, _ in dataloader])

# Store features in .csv file
df = pd.DataFrame(Z.numpy(), index=names)
df.to_csv(f'emdeddings.csv', index=True, index_label='name')
```

### 🔥 Fine-tune RetNeXt

1. Split the data into train, validation and test:

```bash
aidsorb prepare path/to/voxels_data/ --split_ratio='[0.7, 0.15, 0.15]' --seed=42
```

2. Freeze part of the model and fine-tune it:

```python
import torch
from lightning.pytorch import Trainer, seed_everything
from torchmetrics import R2Score, MeanAbsoluteError, MetricCollection
from aidsorb.datamodules import PCDDataModule as VoxelsDataModule
from aidsorb.litmodules import PCDLit as VoxelsLit
from torchvision.transforms.v2 import Compose, RandomChoice
from retnext.modules import RetNeXt
from retnext.transforms import AddChannelDim, BoltzmannFactor

# Freeze parts of the backbone
model = RetNeXt(pretrained=True)
model.backbone[:7].requires_grad_(False)
model.backbone[:7].eval()

# Create the datamodule
datamodule = ...
datamodule.setup()

# Create the trainer
trainer = L.Trainer(...)

# Create the litmodel
loss = torch.nn.MSELoss()

litmodel = VoxelsLit(model=model, )


# Initialize last bias with target mean (optional but recommended)
train_names = list(datamodule.train_dataset.pcd_names)
y_train_mean = datamodule.train_dataset.Y.loc[train_names].mean().item()
torch.nn.init.constant_(model.fc.bias, y_train_mean)

trainer.fit(litmodel, datamodule=datamodule)
```

<details>
<summary>Show RetNeXt architecture</summary>
	
```python
RetNeXt(
  (backbone): Sequential(
    (0): BatchNorm3d(1, eps=1e-05, momentum=None, affine=False, track_running_stats=True)
    (1): Sequential(
      (0): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, bias=False)
      (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (2): Sequential(
      (0): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, bias=False)
      (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (3): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, bias=False)
      (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (5): Sequential(
      (0): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, bias=False)
      (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (6): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Sequential(
      (0): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False)
      (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (8): Sequential(
      (0): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False)
      (1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (9): AdaptiveAvgPool3d(output_size=1)
    (10): Flatten(start_dim=1, end_dim=-1)
  )
  (fc): Linear(in_features=128, out_features=1, bias=True)
)
```
</details>


## 📑 Citing
If you Please use the following BibTeX entry:

```bibtex
@article{Sarikas2024,
  title = {Gas adsorption meets geometric deep learning: points, set and match},
  volume = {14},
  ISSN = {2045-2322},
  url = {http://dx.doi.org/10.1038/s41598-024-76319-8},
  DOI = {10.1038/s41598-024-76319-8},
  number = {1},
  journal = {Scientific Reports},
  publisher = {Springer Science and Business Media LLC},
  author = {Sarikas,  Antonios P. and Gkagkas,  Konstantinos and Froudakis,  George E.},
  year = {2024},
  month = nov
}
```

## ⚖️ License
**RetNeXt** is released under the [GNU General Public License v3.0 only](https://spdx.org/licenses/GPL-3.0-only.html).


