import os
import glob
from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, ConcatItemsd, ToTensord,
    MapLabelValued
)
from monai.data import Dataset, DataLoader

import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

def load_model(checkpoint_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        dropout=0.2,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)

    # strip DataParallel "module." prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "", 1)] = v

    model.load_state_dict(new_state_dict)
    return model


def enable_dropout(model):
    """Keep dropout active during inference for MC sampling."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def get_transforms():
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
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
        ToTensord(keys=["image", "label"]),
    ])


def mc_predict(model, image, n_samples=20, device="cuda"):
    """
    Run N forward passes with dropout active.
    Returns mean prediction and uncertainty map.
    """
    model.eval()
    enable_dropout(model)  # keep dropout on despite eval mode

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = torch.softmax(model(image), dim=1)
            predictions.append(output.cpu().numpy())

    predictions = np.stack(predictions, axis=0)  # (N, 1, 4, H, W, D)

    mean_pred = predictions.mean(axis=0)       # mean across N samples
    uncertainty = predictions.var(axis=0)      # variance = uncertainty

    return mean_pred, uncertainty


def visualise_uncertainty(image, mean_pred, uncertainty, label, save_path="outputs/uncertainty.png"):
    """
    Plot a single axial slice showing:
    - FLAIR input
    - Ground truth
    - Mean prediction
    - Uncertainty map
    """
    # pick middle slice along depth axis
    slice_idx = image.shape[-1] // 2

    flair     = image[0, 3, :, :, slice_idx].cpu().numpy()  # FLAIR channel
    gt        = label[0, 0, :, :, slice_idx].cpu().numpy()
    pred      = mean_pred[0].argmax(axis=0)[:, :, slice_idx]
    unc       = uncertainty[0].max(axis=0)[:, :, slice_idx]  # max uncertainty across classes

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title("FLAIR Input")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="hot", vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="hot", vmin=0, vmax=3)
    axes[2].set_title("Mean Prediction")
    axes[2].axis("off")

    im = axes[3].imshow(unc, cmap="viridis")
    axes[3].set_title("Uncertainty")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3])

    plt.suptitle("Monte Carlo Dropout Uncertainty (N=20 samples)", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model("outputs/model.pth", device)
    print("Model loaded.")

    # get one patient for demo
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    patients = sorted(glob.glob(os.path.join(cfg["data_root"], "BraTS2021_*")))
    patient  = patients[0]
    base     = os.path.basename(patient)

    data_dict = [{
        "t1":    os.path.join(patient, f"{base}_t1.nii.gz"),
        "t1ce":  os.path.join(patient, f"{base}_t1ce.nii.gz"),
        "t2":    os.path.join(patient, f"{base}_t2.nii.gz"),
        "flair": os.path.join(patient, f"{base}_flair.nii.gz"),
        "label": os.path.join(patient, f"{base}_seg.nii.gz"),
    }]

    ds     = Dataset(data=data_dict, transform=get_transforms())
    loader = DataLoader(ds, batch_size=1)
    batch  = next(iter(loader))

    image = batch["image"].to(device)
    label = batch["label"]

    print("Running MC Dropout inference (20 samples)...")
    mean_pred, uncertainty = mc_predict(model, image, n_samples=20, device=device)

    visualise_uncertainty(image, mean_pred, uncertainty, label)
    print("Done.")


if __name__ == "__main__":
    main()