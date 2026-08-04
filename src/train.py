import yaml
import argparse
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, RandSpatialCropd, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
import os
import glob

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 64), random_size=False),
        ToTensord(keys=["image", "label"]),
    ])

def get_data_dicts(data_root):
    patients = sorted(glob.glob(os.path.join(data_root, "BraTS2021_*")))
    data_dicts = []
    for p in patients:
        flair = os.path.join(p, f"{os.path.basename(p)}_flair.nii.gz")
        seg   = os.path.join(p, f"{os.path.basename(p)}_seg.nii.gz")
        if os.path.exists(flair) and os.path.exists(seg):
            data_dicts.append({"image": flair, "label": seg})
    return data_dicts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dicts = get_data_dicts(cfg["data_root"])
    print(f"Found {len(data_dicts)} patients")

    split = int(0.8 * len(data_dicts))
    train_ds = Dataset(data=data_dicts[:split], transform=get_transforms())
    val_ds   = Dataset(data=data_dicts[split:], transform=get_transforms())
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
    ).to(device)

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    metric = DiceMetric(include_background=False, reduction="mean")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            segs = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, segs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "model.pth"))
    print("Model saved.")

if __name__ == "__main__":
    main()