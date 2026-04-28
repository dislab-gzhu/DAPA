#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAPA script for crowd counting models.

This script:
1. Trains an adversarial patch on ShanghaiTech Part A training images.
2. Saves patched adversarial test images.
3. Evaluates transferability across multiple crowd-counting models.

The implementation preserves the original logic where possible, while improving:
- naming,
- structure,
- file handling,
- repeated-code reduction,
- comments and docstrings.
"""

import argparse
import glob
import math
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from utils import *


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DATA_ROOT = Path("/home/Data/Shanghai/")
SAVE_DIR = Path("./dapa")
RESULTS_DIR = Path("./dapa_results")

TRAIN_IMAGE_DIR = DATA_ROOT / "part_A/train_data/images"
TEST_IMAGE_DIR = DATA_ROOT / "part_A/test_data/images"

MODEL_NAMES_FOR_TRANSFER = [
    "mcnn",
    "cannet",
    "bl",
    "dm",
    "sasnet",
    "dgcc",
    "mp",
]

EPOCHS = 2
STEP_SIZE = 0.1
ATTACK_ITERS = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED_TORCH = 1
SEED_NUMPY = 0
SEED_PATCH = 3


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch DAPA attacks for crowd-counting models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mcnn",
        choices=["mcnn", "cannet", "bl", "dm", "sasnet", "dgcc", "mp"],
        help="Source/target model used to train the adversarial patch.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Coefficient for the primary attack loss term.",
    )

    parser.add_argument(
        "--p_size",
        type=int,
        default=81,
        help="Adversarial patch size. Used as square size or circular radius.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Coefficient applied to the negative attribution term.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------

def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(SEED_TORCH)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_TORCH)

    np.random.seed(SEED_NUMPY)


def get_device() -> torch.device:
    """Return CUDA device if available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def discover_images(image_dir: Path) -> List[str]:
    """Return sorted JPEG image paths under the given directory."""
    return sorted(glob.glob(str(image_dir / "*.jpg")))


def get_ground_truth_path(image_path: str) -> str:
    """
    Convert an image path to its corresponding ground-truth HDF5 path.

    Example:
        .../images/IMG_1.jpg
    becomes:
        .../ground_truth/IMG_1.h5
    """
    return image_path.replace(".jpg", ".h5").replace("images", "ground_truth")


def load_density_map(image_path: str) -> np.ndarray:
    """Load the density map corresponding to the given image path."""
    gt_path = get_ground_truth_path(image_path)

    with h5py.File(gt_path, "r") as gt_file:
        density = np.asarray(gt_file["density"])

    return density


def get_image_transform(model_name: str):
    """
    Return the preprocessing transform required by the given model.

    Original behavior:
    - mp: pad image dimensions to multiples of 16
    - mcnn: no normalization
    - others: normalized input
    """
    if model_name == "mp":
        return image_padding_to_16

    if model_name == "mcnn":
        return image_nonprocessing

    return image_normalization


# ---------------------------------------------------------------------
# Patch initialization
# ---------------------------------------------------------------------

def init_patch_square(
    patch_size: int,
    num_patches: int,
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Initialize square random patches.

    Returns:
        patches:
            List of patches. Each patch has shape ``(1, 3, S, S)``.
        patch_shape:
            Shape of one patch.
    """
    np.random.seed(SEED_PATCH)

    patch = np.random.rand(1, 3, patch_size, patch_size).astype(np.float32)
    patches = [patch.copy() for _ in range(num_patches)]

    return patches, patch.shape


def init_patch_circle(
    radius: int,
    num_patches: int,
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Initialize circular random patches.

    The initial canvas has shape ``(1, 3, 2R, 2R)``. Values outside the
    circle are removed by cropping empty rows and columns.

    Returns:
        patches:
            List of circular patches.
        patch_shape:
            Shape of one patch.
    """
    np.random.seed(SEED_PATCH)

    patch = np.zeros((1, 3, radius * 2, radius * 2), dtype=np.float32)

    y, x = np.ogrid[-radius:radius, -radius:radius]
    circle_mask = x**2 + y**2 <= radius**2

    for channel in range(3):
        channel_patch = np.zeros((radius * 2, radius * 2), dtype=np.float32)
        channel_patch[circle_mask] = np.random.rand()

        empty_rows = np.flatnonzero((channel_patch == 0).all(axis=1))
        channel_patch = np.delete(channel_patch, empty_rows, axis=0)

        empty_cols = np.flatnonzero((channel_patch == 0).all(axis=0))
        channel_patch = np.delete(channel_patch, empty_cols, axis=1)

        patch[0, channel] = channel_patch

    patches = [patch.copy() for _ in range(num_patches)]

    return patches, patch.shape


def extract_effective_patches(
    patch_tensor: Tensor,
    mask_tensor: Tensor,
    patch_shape: Tuple[int, int, int, int],
) -> List[np.ndarray]:
    """
    Extract tight cropped patches from the masked patch tensor.

    This mirrors the original behavior:
    ``masked_patch = mask * patch``, then each channel is passed through
    ``submatrix`` to remove unused rows/columns.
    """
    new_patches: List[np.ndarray] = []

    for patch_index in range(len(patch_tensor)):
        masked_patch = (
            mask_tensor[patch_index] * patch_tensor[patch_index]
        ).detach().cpu().numpy()

        new_patch = np.zeros(patch_shape, dtype=np.float32)

        for batch_index in range(new_patch.shape[0]):
            for channel in range(new_patch.shape[1]):
                new_patch[batch_index, channel] = submatrix(
                    masked_patch[batch_index, channel]
                )

        new_patches.append(new_patch)

    return new_patches


# ---------------------------------------------------------------------
# Core attack logic
# ---------------------------------------------------------------------

def train_dcm(
    model: nn.Module,
    model_name: str,
    train_image_paths: List[str],
    patch_size: int,
    beta: float,
    gamma: float,
    device: torch.device,
    criterion_reg: nn.Module,
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Train adversarial patches on the training set.

    Args:
        model:
            Source model being attacked.
        model_name:
            Name of the source model.
        train_image_paths:
            Training image paths.
        patch_size:
            Initial patch size.
        beta:
            Coefficient for the attack loss term.
        gamma:
            Coefficient for negative attribution.
        device:
            Torch device.
        criterion_reg:
            Regression criterion used in the attack.

    Returns:
        patch:
            Learned adversarial patch list.
        patch_shape:
            Shape of each patch.
    """
    model.eval()
    image_transform = get_image_transform(model_name)

    patch: List[np.ndarray] = []
    patch_shape: Tuple[int, int, int, int] = (0, 0, 0, 0)

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"train: epoch {epoch + 1}")

        for image_path in tqdm(train_image_paths):
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_transform(image).to(device)

            ground_truth = load_density_map(image_path)

            # Shape of model input after batch dimension is added.
            data_shape = image_tensor.unsqueeze(0).detach().cpu().numpy().shape

            # Initialize patch only once, at the first training image.
            if epoch == 0 and not patch:
                patch, patch_shape = init_patch_square(
                    patch_size=patch_size,
                    num_patches=1,
                )

                # If you want the original circular patch behavior instead,
                # use this and comment out init_patch_square above:
                #
                # patch, patch_shape = init_patch_circle(
                #     radius=patch_size,
                #     num_patches=1,
                # )

            transformed_patch, transformed_mask = patch_transform(
                patch,
                data_shape,
                patch_shape,
                (image_tensor.shape[1], image_tensor.shape[2]),
                num_patch=1,
            )

            patch_tensor = torch.as_tensor(
                np.array(transformed_patch),
                dtype=torch.float32,
                device=device,
            )

            mask_tensor = torch.as_tensor(
                np.array(transformed_mask),
                dtype=torch.float32,
                device=device,
            )

            _, mask_tensor, patch_tensor = attack(
                model=model,
                x=image_tensor.unsqueeze(0),
                ground_truth=ground_truth,
                patch=patch_tensor,
                mask=mask_tensor,
                beta=beta,
                gamma=gamma,
                device=device,
                criterion_reg=criterion_reg,
                iters=ATTACK_ITERS,
            )

            patch = extract_effective_patches(
                patch_tensor=patch_tensor,
                mask_tensor=mask_tensor,
                patch_shape=patch_shape,
            )

    elapsed = time.time() - start_time
    avg_time_per_image = elapsed / max(1, len(train_image_paths) * EPOCHS)

    print(f"Training finished in {elapsed:.2f}s")
    print(f"Average time per image: {avg_time_per_image:.4f}s")

    return patch, patch_shape


def attack(
    model: nn.Module,
    x: Tensor,
    ground_truth: np.ndarray,
    patch: Tensor,
    mask: Tensor,
    beta: float,
    gamma: float,
    device: torch.device,
    criterion_reg: nn.Module,
    iters: int = 25,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Optimize adversarial patch values using gradients through the input image.

    Args:
        model:
            Crowd-counting model to attack.
        x:
            Input image tensor with shape ``(1, C, H, W)`` and values in ``[0, 1]``.
        ground_truth:
            Ground-truth density map as a NumPy array.
        patch:
            Patch tensor with shape similar to ``(num_patches, 1, 3, H, W)``
            after transformation.
        mask:
            Binary mask tensor with the same broadcastable shape as ``patch``.
        beta:
            Coefficient for the attack loss term.
        gamma:
            Coefficient for the negative attribution term.
        device:
            Torch device.
        criterion_reg:
            Regression loss criterion.
        iters:
            Number of gradient update iterations.

    Returns:
        adv_x:
            Final adversarial image tensor.
        mask:
            Patch mask tensor.
        patch:
            Updated patch tensor.
    """
    model.eval()

    adv_x = apply_patch(x, patch, mask)
    adv_x.requires_grad_(True)

    x_out = forward_count_model(model, adv_x)

    target = resize_density_to_output(
        density=ground_truth,
        output_height=x_out.shape[2],
        output_width=x_out.shape[3],
        device=device,
    )

    for _ in range(iters):
        model.zero_grad()

        loss = compute_attack_loss(
            prediction=x_out,
            target=target,
            beta=beta,
            gamma=gamma,
            criterion_reg=criterion_reg,
        )

        loss.backward()

        adv_grad = adv_x.grad.clone()

        # Update patch values using gradients with respect to adv_x.
        for patch_index in range(len(patch)):
            patch[patch_index] = patch[patch_index] + STEP_SIZE * adv_grad
            adv_x = (1 - mask[patch_index]) * adv_x + mask[patch_index] * patch[patch_index]

        adv_x = torch.clamp(adv_x, 0, 1)

        # Prepare for the next iteration.
        adv_x = adv_x.detach().requires_grad_(True)
        x_out = forward_count_model(model, adv_x)

    return adv_x.detach(), mask, patch.detach()


def apply_patch(x: Tensor, patch: Tensor, mask: Tensor) -> Tensor:
    """
    Apply one or more transformed patches to an input image tensor.
    """
    adv_x = x

    for patch_index in range(patch.shape[0]):
        adv_x = (1 - mask[patch_index]) * adv_x + mask[patch_index] * patch[patch_index]

    return torch.clamp(adv_x, 0, 1)


def forward_count_model(model: nn.Module, image_tensor: Tensor) -> Tensor:
    """
    Forward pass through the model using ImageNet normalization.

    This preserves the original attack behavior, which normalized the
    adversarial image before model inference.
    """
    image_tensor = batch_norm(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
    return model(image_tensor)


def resize_density_to_output(
    density: np.ndarray,
    output_height: int,
    output_width: int,
    device: torch.device,
) -> Tensor:
    """
    Resize a density map to match model output dimensions.
    """
    resized_density = cv2.resize(
        density,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return (
        torch.from_numpy(resized_density)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


def compute_attack_loss(
    prediction: Tensor,
    target: Tensor,
    beta: float,
    gamma: float,
    criterion_reg: nn.Module,
) -> Tensor:
    """
    Compute the DAPA attack loss.

    Original logic:
    - Build a sign-based weight map.
    - Split prediction into positive and negative components.
    - Combine attribution terms.
    - Add L1 density regression loss.
    """
    weight = torch.sign(torch.abs(target - prediction))

    blank = torch.zeros_like(prediction)
    positive_attr = torch.where(prediction >= 0, prediction, blank)
    negative_attr = torch.where(prediction < 0, prediction, blank)

    balanced_attr = positive_attr + gamma * negative_attr

    attack_term = beta * torch.sum(weight.data * balanced_attr)
    regression_term = criterion_reg(target, prediction)

    return attack_term + regression_term


# ---------------------------------------------------------------------
# Saving adversarial images
# ---------------------------------------------------------------------

def save_adversarial_test_images(
    patch: List[np.ndarray],
    patch_shape: Tuple[int, int, int, int],
    test_image_paths: List[str],
    device: torch.device,
    source_model_name: str,
) -> None:
    """
    Apply the learned patch to the test set and save adversarial images.
    """
    print(f"Saving adversarial images for source model: {source_model_name}")

    for image_path in tqdm(test_image_paths):
        image = Image.open(image_path).convert("RGB")

        # Original behavior used image_nonprocessing here regardless of model.
        image_tensor = image_nonprocessing(image).to(device)

        data_shape = image_tensor.unsqueeze(0).detach().cpu().numpy().shape

        transformed_patch, transformed_mask = patch_transform(
            patch,
            data_shape,
            patch_shape,
            (image_tensor.shape[1], image_tensor.shape[2]),
            num_patch=1,
        )

        patch_tensor = torch.as_tensor(
            np.array(transformed_patch),
            dtype=torch.float32,
            device=device,
        )

        mask_tensor = torch.as_tensor(
            np.array(transformed_mask),
            dtype=torch.float32,
            device=device,
        )

        adv_x = apply_patch(
            x=image_tensor.unsqueeze(0),
            patch=patch_tensor,
            mask=mask_tensor,
        )

        save_path = SAVE_DIR / Path(image_path).name

        # adv_x has shape (1, C, H, W), so save each image in the batch.
        for image_in_batch in adv_x:
            adv_image = transforms.ToPILImage()(image_in_batch.detach().cpu())
            adv_image.save(save_path)

        # Preserve original behavior: update the patch crop after each test image.
        patch = extract_effective_patches(
            patch_tensor=patch_tensor,
            mask_tensor=mask_tensor,
            patch_shape=patch_shape,
        )

    print("Saving done!")


# ---------------------------------------------------------------------
# Cross-model evaluation
# ---------------------------------------------------------------------

def estimate_cross_model(
    source_model_name: str,
    patch_size: int,
    beta: float,
    gamma: float,
    test_image_paths: List[str],
    device: torch.device,
) -> None:
    """
    Evaluate saved adversarial images across target models.

    Results are appended to:
        ./dapa_results/{patch_size}_dcm_{source_model}_{beta}_results(A).txt
    """
    log_path = RESULTS_DIR / (
        f"{patch_size}_dcm_{source_model_name}_{beta}_results(A).txt"
    )

    for target_model_name in MODEL_NAMES_FOR_TRANSFER:
        print(
            f"DCM transfer evaluation: "
            f"source={source_model_name}, target={target_model_name}"
        )

        target_model = load_model_A(target_model_name, device).eval()
        image_transform = get_image_transform(target_model_name)

        mae, rmse = evaluate_model_on_adversarial_images(
            model=target_model,
            model_name=target_model_name,
            image_transform=image_transform,
            test_image_paths=test_image_paths,
            device=device,
        )

        print(f" * MAE  {mae:.2f}")
        print(f" * RMSE {rmse:.2f}")

        append_cross_model_result(
            log_path=log_path,
            source_model_name=source_model_name,
            target_model_name=target_model_name,
            mae=mae,
            rmse=rmse,
        )


def evaluate_model_on_adversarial_images(
    model: nn.Module,
    model_name: str,
    image_transform,
    test_image_paths: List[str],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute MAE and RMSE for a model on saved adversarial test images.
    """
    total_abs_error = 0.0
    total_squared_error = 0.0

    for image_path in test_image_paths:
        adv_image_path = SAVE_DIR / Path(image_path).name

        image = Image.open(adv_image_path).convert("RGB")
        image_tensor = image_transform(image).to(device)

        ground_truth = load_density_map(image_path)

        adv_x = image_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(adv_x)

        gt_count = float(ground_truth.sum())
        pred_count = output.detach().sum()

        if model_name in {"dgcc", "sasnet", "mp"}:
            pred_count = pred_count / 1000.0

        pred_count = float(pred_count.item())

        error = gt_count - pred_count

        total_abs_error += abs(error)
        total_squared_error += error * error

    num_images = max(1, len(test_image_paths))
    mae = total_abs_error / num_images
    rmse = math.sqrt(total_squared_error / num_images)

    return mae, rmse


def append_cross_model_result(
    log_path: Path,
    source_model_name: str,
    target_model_name: str,
    mae: float,
    rmse: float,
) -> None:
    """
    Append one cross-model evaluation result to the log file.
    """
    if not log_path.exists():
        log_path.write_text(
            "source_model\ttarget_model\tmae\trmse\n",
            encoding="utf-8",
        )

    line = f"{source_model_name}\t{target_model_name}\t{mae:.1f}\t{rmse:.1f}\n"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(line)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    set_random_seeds()
    ensure_output_dirs()

    device = get_device()

    train_image_paths = discover_images(TRAIN_IMAGE_DIR)
    test_image_paths = discover_images(TEST_IMAGE_DIR)

    if not train_image_paths:
        raise FileNotFoundError(f"No training images found in: {TRAIN_IMAGE_DIR}")

    if not test_image_paths:
        raise FileNotFoundError(f"No test images found in: {TEST_IMAGE_DIR}")

    print(
        "Part A: Starting DAPA attack "
        f"source_model={args.model}, "
        f"patch_size={args.p_size}, "
        f"beta={args.beta}, "
        f"gamma={args.gamma}, "
        f"device={device}"
    )

    model = load_model_A(args.model, device)
    criterion_reg = nn.L1Loss()

    patch, patch_shape = train_dcm(
        model=model,
        model_name=args.model,
        train_image_paths=train_image_paths,
        patch_size=args.p_size,
        beta=args.beta,
        gamma=args.gamma,
        device=device,
        criterion_reg=criterion_reg,
    )

    save_adversarial_test_images(
        patch=patch,
        patch_shape=patch_shape,
        test_image_paths=test_image_paths,
        device=device,
        source_model_name=args.model,
    )

    estimate_cross_model(
        source_model_name=args.model,
        patch_size=args.p_size,
        beta=args.beta,
        gamma=args.gamma,
        test_image_paths=test_image_paths,
        device=device,
    )


if __name__ == "__main__":
    main()
