from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


PATH_NAME = './models/UNET5_mini_dataset'
rootname = './'

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0

data_dir = './data/processed/'

train_images = sorted(
    glob.glob(os.path.join(data_dir, "contrastEnhanced", "*.png")))

train_labels = sorted(
    glob.glob(os.path.join(data_dir, "grayLabels", "*.png")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files, test_files = data_dicts[:3], data_dicts[3:4], data_dicts[4:]

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
)

test_ds = CacheDataset(
    data=test_files, transform=val_transforms, cache_rate=.5, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model2 = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=4,
    norm=Norm.BATCH,
).to(device)

temp = torch.load(PATH_NAME)
model.load_state_dict(temp['model_state_dict'])

dice_metric = DiceMetric(include_background=False, reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

temp = torch.load(PATH_NAME)
model.load_state_dict(temp['model_state_dict'])
optimizer.load_state_dict(temp['optimizer_state_dict'])
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(test_loader):
#         roi_size = (160, 160)
        roi_size = (256, 256)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 18))
        plt.title(f"image {i}")
        plt.imshow(np.transpose(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :]),  cmap="gray")
        plt.savefig(rootname + 'segmentation')
        if i == 0:
            break