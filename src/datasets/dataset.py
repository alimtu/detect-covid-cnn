"""Chest X-ray dataset: discovery, sampling, caching, stratified split, loaders.

The loader is decoupled from any particular on-disk layout: for each class
directory it uses the ``images/`` subfolder if present, otherwise reads image
files directly from the class directory. Any sibling ``masks/`` folder is
ignored (mask support is reserved for future segmentation work).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.datasets.transforms import build_transforms

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DataBundle:
    """Prepared datasets plus metadata used downstream."""

    train: "ChestXrayDataset"
    val: "ChestXrayDataset"
    test: "ChestXrayDataset"
    class_names: list[str]
    train_targets: list[int] = field(default_factory=list)


class ChestXrayDataset(Dataset):
    """Dataset over a list of ``(image_path, label)`` samples.

    Args:
        samples: List of ``(path, label)`` pairs.
        transform: Optional transform applied to each PIL image.
        cache_images: If ``True``, decoded RGB images are cached in memory
            (useful for small datasets; the transform is still applied per read).
    """

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        transform: Callable | None = None,
        cache_images: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.cache_images = cache_images
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, index: int) -> Image.Image:
        if self.cache_images and index in self._cache:
            return self._cache[index]
        path, _ = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.cache_images:
            self._cache[index] = image
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        _, label = self.samples[index]
        image = self._load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _class_image_dir(class_dir: Path) -> Path:
    """Return the directory holding a class's images (auto-detects ``images/``)."""
    nested = class_dir / "images"
    return nested if nested.is_dir() else class_dir


def _list_images(directory: Path) -> list[Path]:
    """Return sorted image files within ``directory`` (non-recursive)."""
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    return sorted(files)


def discover_class_names(dataset_path: str | Path) -> list[str]:
    """Return sorted class names (subfolders) for a dataset root.

    Args:
        dataset_path: Root directory containing one subfolder per class.

    Returns:
        Alphabetically sorted class names.

    Raises:
        FileNotFoundError: If the dataset path or class subfolders are missing.
    """
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class subfolders found in: {root}")
    return class_names


def discover_samples(
    dataset_path: str | Path,
    sample_per_class: int | str,
    seed: int,
    class_names: Sequence[str] | None = None,
) -> tuple[list[tuple[Path, int]], list[str]]:
    """Discover class folders and build a (optionally sampled) sample list.

    Args:
        dataset_path: Root directory containing one subfolder per class.
        sample_per_class: Maximum images to keep per class, or ``"all"``.
        seed: Seed controlling which images are sampled per class.
        class_names: Optional explicit class order; auto-discovered if ``None``.

    Returns:
        ``(samples, class_names)`` where ``samples`` is a list of
        ``(path, label)`` pairs.

    Raises:
        FileNotFoundError: If the dataset path or class images are missing.
    """
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    names = list(class_names) if class_names else discover_class_names(root)

    use_all = isinstance(sample_per_class, str) and sample_per_class.lower() == "all"
    limit = None if use_all else int(sample_per_class)

    rng = np.random.default_rng(seed)
    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(names):
        image_dir = _class_image_dir(root / class_name)
        images = _list_images(image_dir)
        if not images:
            raise FileNotFoundError(f"No images found for class '{class_name}' in {image_dir}")
        if limit is not None and limit < len(images):
            chosen = rng.choice(len(images), size=limit, replace=False)
            images = [images[i] for i in sorted(chosen)]
        samples.extend((path, label) for path in images)

    return samples, names


def _stratified_split(
    samples: Sequence[tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Split samples into train/val/test preserving class proportions."""
    labels = [label for _, label in samples]
    train_val, test = train_test_split(
        list(samples), test_size=test_ratio, stratify=labels, random_state=seed
    )
    train_val_labels = [label for _, label in train_val]
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val_labels, random_state=seed
    )
    return train, val, test


def compute_class_weights(
    targets: Sequence[int],
    num_classes: int,
    spec: object,
) -> torch.Tensor | None:
    """Resolve class weights from a config spec.

    Args:
        targets: Training labels used to estimate frequencies.
        num_classes: Number of classes.
        spec: ``None`` (uniform -> returns ``None``), ``"balanced"`` (inverse
            frequency) or an explicit list of per-class weights.

    Returns:
        A float tensor of shape ``(num_classes,)`` or ``None`` for uniform.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.lower() != "balanced":
            raise ValueError(f"Unknown class_weights spec '{spec}'. Use null|balanced|list.")
        counts = np.zeros(num_classes, dtype=np.float64)
        for label in targets:
            counts[label] += 1
        counts = np.clip(counts, a_min=1.0, a_max=None)
        weights = counts.sum() / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)
    # Explicit list.
    weights = list(spec)
    if len(weights) != num_classes:
        raise ValueError(
            f"class_weights list has {len(weights)} entries, expected {num_classes}."
        )
    return torch.tensor([float(w) for w in weights], dtype=torch.float32)


def build_dataloaders(config) -> tuple[DataBundle, dict[str, DataLoader]]:
    """Build datasets and dataloaders from configuration.

    Args:
        config: The full experiment configuration.

    Returns:
        ``(DataBundle, dataloaders)`` where ``dataloaders`` maps ``"train"``,
        ``"val"`` and ``"test"`` to :class:`DataLoader` objects.
    """
    ds_cfg = config["dataset"]
    seed = int(config["reproducibility"]["seed"])
    class_names_cfg = config.get("dataset.class_names")

    samples, class_names = discover_samples(
        dataset_path=ds_cfg["path"],
        sample_per_class=ds_cfg["sample_per_class"],
        seed=seed,
        class_names=list(class_names_cfg) if class_names_cfg else None,
    )

    train_samples, val_samples, test_samples = _stratified_split(
        samples,
        train_ratio=float(ds_cfg["split"]["train"]),
        val_ratio=float(ds_cfg["split"]["val"]),
        test_ratio=float(ds_cfg["split"]["test"]),
        seed=seed,
    )

    train_transform, eval_transform = build_transforms(config)
    cache = bool(ds_cfg.get("cache_images", False))

    bundle = DataBundle(
        train=ChestXrayDataset(train_samples, train_transform, cache_images=cache),
        val=ChestXrayDataset(val_samples, eval_transform, cache_images=cache),
        test=ChestXrayDataset(test_samples, eval_transform, cache_images=cache),
        class_names=class_names,
        train_targets=[label for _, label in train_samples],
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 0))
    pin_memory = bool(ds_cfg.get("pin_memory", False))
    shuffle = bool(ds_cfg.get("shuffle", True))

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            bundle.train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        "val": DataLoader(
            bundle.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            bundle.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return bundle, dataloaders
