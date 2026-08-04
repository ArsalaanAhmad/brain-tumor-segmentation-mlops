import yaml
import argparse
import torch
import wandb
import os
import glob
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, RandSpatialCropd, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(128, 128, 64),
            random_size=False
        ),
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

    data_dicts = get_data_dicts(cfg["data_root"])
    print(f"Found {len(data_dicts)} patients")

    if not data_dicts:
        raise FileNotFoundError(
            f"No BraTS samples were found under {cfg['data_root']}. "
            "Update configs/baseline.yaml or pass a valid data_root."
        )

    # Initialise W&B after we know the dataset path is valid.
    wandb.init(
        project="brain-tumor-segmentation-mlops",
        config=cfg
    )

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split = max(1, int(0.8 * len(data_dicts)))
    train_ds = Dataset(data=data_dicts[:split], transform=get_transforms())
    val_ds   = Dataset(data=data_dicts[split:],  transform=get_transforms())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=1)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        dropout=0.2,  # needed for Monte Carlo Dropout later
    ).to(device)

    loss_fn   = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"]
    )
    metric    = DiceMetric(include_background=False, reduction="mean")

    for epoch in range(cfg["training"]["epochs"]):
        # --- training ---
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

        avg_loss = epoch_loss / len(train_loader)

        # --- validation ---
        model.eval()
        if len(val_ds) > 0:
            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs = val_batch["image"].to(device)
                    val_segs = val_batch["label"].to(device)
                    val_outputs = model(val_imgs)
                    metric(y_pred=val_outputs, y=val_segs)
            dice_score = metric.aggregate().item()
            metric.reset()
        else:
            dice_score = float("nan")

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Dice: {dice_score:.4f}")

        # log to W&B — this is what shows up in your dashboard
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_dice": dice_score,
        })

    # --- save model ---
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["output_dir"], "model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # --- ONNX export ---
    dummy_input = torch.randn(1, 4, 128, 128, 64).to(device)
    onnx_path   = os.path.join(cfg["output_dir"], "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["mri_input"],
        output_names=["segmentation"],
        opset_version=14
    )
    print(f"ONNX model exported to {onnx_path}")
    wandb.save(onnx_path)

    wandb.finish()


if __name__ == "__main__":
    main()