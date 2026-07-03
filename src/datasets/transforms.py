"""Build train/eval image transforms from the augmentation configuration.

Every augmentation is optional and parameterised by config. An optional ``p``
on a transform wraps it in :class:`~torchvision.transforms.RandomApply` so it
fires with that probability. Evaluation always uses a deterministic
resize + normalize pipeline (no augmentation).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from torchvision import transforms

# Fallback normalization if the config disables/omits it (identity-like resize).
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _enabled(section: Mapping[str, Any] | None) -> bool:
    return bool(section) and bool(section.get("enabled", False))


def _maybe_random_apply(transform: Callable, prob: float | None) -> Callable:
    """Wrap ``transform`` in RandomApply when ``0 <= prob < 1``."""
    if prob is not None and prob < 1.0:
        return transforms.RandomApply([transform], p=float(prob))
    return transform


def _to_plain(value: Any) -> Any:
    """Convert a Config/mapping to a plain dict for uniform access."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _sizing_transforms(aug: Mapping[str, Any], image_size: int, train: bool) -> list:
    """Return the leading resize/crop transforms for train or eval."""
    rrc = _to_plain(aug.get("random_resized_crop"))
    rc = _to_plain(aug.get("random_crop"))

    if train and _enabled(rrc):
        return [
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(rrc.get("scale", (0.8, 1.0))),
                ratio=tuple(rrc.get("ratio", (0.75, 1.3333))),
            )
        ]
    if train and _enabled(rc):
        return [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=int(rc.get("padding", 0))),
        ]
    return [transforms.Resize((image_size, image_size))]


def _augmentation_transforms(aug: Mapping[str, Any]) -> list:
    """Return the (order-sensitive) list of enabled train-time augmentations."""
    ops: list = []

    hflip = _to_plain(aug.get("random_horizontal_flip"))
    if _enabled(hflip):
        ops.append(transforms.RandomHorizontalFlip(p=float(hflip.get("p", 0.5))))

    vflip = _to_plain(aug.get("random_vertical_flip"))
    if _enabled(vflip):
        ops.append(transforms.RandomVerticalFlip(p=float(vflip.get("p", 0.5))))

    rot = _to_plain(aug.get("random_rotation"))
    if _enabled(rot):
        ops.append(
            _maybe_random_apply(
                transforms.RandomRotation(degrees=float(rot.get("degrees", 10))),
                rot.get("p"),
            )
        )

    affine = _to_plain(aug.get("random_affine"))
    if _enabled(affine):
        ops.append(
            _maybe_random_apply(
                transforms.RandomAffine(
                    degrees=float(affine.get("degrees", 0)),
                    translate=tuple(affine["translate"]) if affine.get("translate") else None,
                    scale=tuple(affine["scale"]) if affine.get("scale") else None,
                    shear=affine.get("shear", 0),
                ),
                affine.get("p"),
            )
        )

    jitter = _to_plain(aug.get("color_jitter"))
    if _enabled(jitter):
        ops.append(
            _maybe_random_apply(
                transforms.ColorJitter(
                    brightness=float(jitter.get("brightness", 0.0)),
                    contrast=float(jitter.get("contrast", 0.0)),
                    saturation=float(jitter.get("saturation", 0.0)),
                    hue=float(jitter.get("hue", 0.0)),
                ),
                jitter.get("p"),
            )
        )

    blur = _to_plain(aug.get("gaussian_blur"))
    if _enabled(blur):
        ops.append(
            _maybe_random_apply(
                transforms.GaussianBlur(
                    kernel_size=int(blur.get("kernel_size", 3)),
                    sigma=tuple(blur.get("sigma", (0.1, 2.0))),
                ),
                blur.get("p"),
            )
        )
    return ops


def _normalize_transform(aug: Mapping[str, Any]) -> Callable | None:
    norm = _to_plain(aug.get("normalize"))
    if not _enabled(norm):
        return None
    return transforms.Normalize(
        mean=tuple(norm.get("mean", DEFAULT_MEAN)),
        std=tuple(norm.get("std", DEFAULT_STD)),
    )


def build_transforms(config) -> tuple[Callable, Callable]:
    """Build ``(train_transform, eval_transform)`` from the configuration.

    Args:
        config: The full experiment configuration.

    Returns:
        A tuple of composed train and eval transforms.
    """
    aug = _to_plain(config["augmentation"])
    image_size = int(config["dataset"]["image_size"])
    normalize = _normalize_transform(aug)

    train_ops = _sizing_transforms(aug, image_size, train=True)
    train_ops += _augmentation_transforms(aug)
    train_ops.append(transforms.ToTensor())
    if normalize is not None:
        train_ops.append(normalize)

    eval_ops: list = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize is not None:
        eval_ops.append(normalize)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_eval_transform(image_size: int, mean, std) -> Callable:
    """Build a deterministic eval transform for inference from explicit params.

    Args:
        image_size: Target square size.
        mean: Per-channel normalization means.
        std: Per-channel normalization stds.

    Returns:
        The composed evaluation transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
