from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureType,
    RandAdjustContrastd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.nets import BasicUnet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

max_epochs = 200
val_interval = 2

modelname = 'UNET5_mini_dataset'
rootname = './'
PATH_NAME =  os.path.join(rootname, 'models', modelname)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0

# rootname = '/zfsauton2/home/eerickson/16725/'
data_dir =  os.path.join(rootname, 'data', 'processed')

train_images = sorted(
    glob.glob(os.path.join(data_dir, "centerFrames", "*.png")))

train_labels = sorted(
    glob.glob(os.path.join(data_dir, "grayLabels", "*.png")))

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files, test_files = data_dicts[:3], data_dicts[3:4], data_dicts[4:]

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=255,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.5, 4.5)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
#             spatial_size=(96, 96),
            spatial_size=(512, 512),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
    ]
)
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

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=1)

print(train_files)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=.5, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

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
    channels=(8, 32, 64, 128, 256, 512, 1024),
    strides=(4, 2, 2, 2, 2, 2),
    num_res_units=4,
    norm=Norm.BATCH,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5) #change learning rate here
dice_metric = DiceMetric(include_background=False, reduction="mean")


best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
#         print(np.shape(inputs))
#         print(np.shape(labels))
        
        optimizer.zero_grad()
        outputs = model(inputs)
#         print(np.shape(outputs))
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(
        #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #     f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
#                 roi_size = (160, 160)
                roi_size = (256,256)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save({
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
          }, PATH_NAME)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )