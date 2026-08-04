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
    NormalizeIntensityd, RandSpatialCropd,
    ToTensord, ConcatItemsd, MapLabelValued
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_transforms():
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
        # remap BraTS label 4 → 3 (labels are 0,1,2,4 — not 0,1,2,3)
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3]
        ),
        NormalizeIntensityd(
            keys=["t1", "t1ce", "t2", "flair"],
            nonzero=True,
            channel_wise=True
        ),
        ConcatItemsd(keys=["t1", "t1ce", "t2", "flair"], name="image"),
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
        base  = os.path.basename(p)
        t1    = os.path.join(p, f"{base}_t1.nii.gz")
        t1ce  = os.path.join(p, f"{base}_t1ce.nii.gz")
        t2    = os.path.join(p, f"{base}_t2.nii.gz")
        flair = os.path.join(p, f"{base}_flair.nii.gz")
        seg   = os.path.join(p, f"{base}_seg.nii.gz")
        if all(os.path.exists(f) for f in [t1, t1ce, t2, flair, seg]):
            data_dicts.append({
                "t1": t1, "t1ce": t1ce,
                "t2": t2, "flair": flair,
                "label": seg
            })
    return data_dicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only N patients for quick testing")
    args = parser.parse_args()
    cfg = load_config(args.config)

    wandb.init(
        project="brain-tumor-segmentation-mlops",
        config=cfg
    )

    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dicts = get_data_dicts(cfg["data_root"])
    
    # subset for fast iteration/testing
    if args.subset:
        data_dicts = data_dicts[:args.subset]
        print(f"Using subset of {args.subset} patients")
    else:
        print(f"Using full dataset: {len(data_dicts)} patients")

    split        = int(0.8 * len(data_dicts))
    train_ds     = Dataset(data=data_dicts[:split], transform=get_transforms())
    val_ds       = Dataset(data=data_dicts[split:],  transform=get_transforms())
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    val_loader   = DataLoader(val_ds, batch_size=1)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        dropout=0.2,
    ).to(device)

    loss_fn   = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"]
    )
    metric = DiceMetric(include_background=False, reduction="mean")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            segs = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, segs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print every batch so you can see live progress
            print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_imgs = val_batch["image"].to(device)
                val_segs = val_batch["label"].to(device)
                val_out  = model(val_imgs)
                metric(y_pred=val_out, y=val_segs)
        dice_score = metric.aggregate().item()
        metric.reset()

        print(f"✅ Epoch {epoch+1} COMPLETE | Loss: {avg_loss:.4f} | Dice: {dice_score:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_dice": dice_score,
        })

    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["output_dir"], "model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    dummy     = torch.randn(1, 4, 128, 128, 64).to(device)
    onnx_path = os.path.join(cfg["output_dir"], "model.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["mri_input"],
        output_names=["segmentation"],
        opset_version=14
    )
    print(f"ONNX exported to {onnx_path}")
    wandb.save(onnx_path)
    wandb.finish()


if __name__ == "__main__":
    main()