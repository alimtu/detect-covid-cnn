"""Transfer-learning model factory.

A registry maps a config name to a builder that (a) loads a torchvision
backbone with optional pretrained weights and (b) replaces its classifier head
with one sized for ``num_classes`` (optionally with dropout and a hidden layer).

Adding a new architecture means registering one builder function; nothing else
in the pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torchvision import models


@dataclass
class ModelBundle:
    """A model together with a handle to its (replaceable) classifier head."""

    model: nn.Module
    head: nn.Module


def make_classifier(
    in_features: int,
    num_classes: int,
    dropout: float = 0.0,
    hidden: int | None = None,
) -> nn.Module:
    """Build a classifier head.

    Args:
        in_features: Number of input features from the backbone.
        num_classes: Number of output classes.
        dropout: Dropout probability applied before the final layer.
        hidden: Optional hidden layer size (adds ``Linear -> ReLU``).

    Returns:
        The classifier module.
    """
    layers: list[nn.Module] = []
    features = in_features
    if hidden:
        layers += [nn.Linear(in_features, hidden), nn.ReLU(inplace=True)]
        features = hidden
    if dropout and dropout > 0.0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(features, num_classes))
    return layers[0] if len(layers) == 1 else nn.Sequential(*layers)


def _build_densenet121(num_classes: int, pretrained: bool, dropout: float, hidden: int | None) -> ModelBundle:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    head = make_classifier(model.classifier.in_features, num_classes, dropout, hidden)
    model.classifier = head
    return ModelBundle(model, head)


def _build_vgg16(num_classes: int, pretrained: bool, dropout: float, hidden: int | None) -> ModelBundle:
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)
    head = make_classifier(model.classifier[-1].in_features, num_classes, dropout, hidden)
    model.classifier[-1] = head
    return ModelBundle(model, head)


def _build_resnet50(num_classes: int, pretrained: bool, dropout: float, hidden: int | None) -> ModelBundle:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    head = make_classifier(model.fc.in_features, num_classes, dropout, hidden)
    model.fc = head
    return ModelBundle(model, head)


def _build_efficientnet_b0(num_classes: int, pretrained: bool, dropout: float, hidden: int | None) -> ModelBundle:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # classifier is Sequential(Dropout, Linear); replace the final Linear.
    head = make_classifier(model.classifier[-1].in_features, num_classes, dropout, hidden)
    model.classifier[-1] = head
    return ModelBundle(model, head)


def _build_vit_b_16(num_classes: int, pretrained: bool, dropout: float, hidden: int | None) -> ModelBundle:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    head = make_classifier(model.heads.head.in_features, num_classes, dropout, hidden)
    model.heads.head = head
    return ModelBundle(model, head)


_MODEL_REGISTRY: dict[str, Callable[[int, bool, float, int | None], ModelBundle]] = {
    "densenet121": _build_densenet121,
    "vgg16": _build_vgg16,
    "resnet50": _build_resnet50,
    "efficientnet_b0": _build_efficientnet_b0,
    "vit_b_16": _build_vit_b_16,
}


def available_models() -> list[str]:
    """Return the list of supported model names."""
    return sorted(_MODEL_REGISTRY)


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    classifier_hidden: int | None = None,
) -> ModelBundle:
    """Build a model by name.

    Args:
        name: Model identifier (see :func:`available_models`).
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained ImageNet weights.
        dropout: Dropout probability in the classifier head.
        classifier_hidden: Optional hidden layer size in the head.

    Returns:
        A :class:`ModelBundle` with the model and its classifier head.

    Raises:
        ValueError: If ``name`` is not a supported model.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(available_models())}")
    return _MODEL_REGISTRY[key](num_classes, pretrained, dropout, classifier_hidden)


def build_model_from_config(model_cfg, num_classes: int) -> ModelBundle:
    """Build a model from the ``model`` config section.

    Args:
        model_cfg: The ``model`` configuration section.
        num_classes: Number of output classes.

    Returns:
        A :class:`ModelBundle`.
    """
    return build_model(
        name=str(model_cfg["name"]),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0) or 0.0),
        classifier_hidden=model_cfg.get("classifier_hidden"),
    )


def freeze_backbone(bundle: ModelBundle) -> None:
    """Freeze all parameters except the classifier head."""
    for param in bundle.model.parameters():
        param.requires_grad = False
    for param in bundle.head.parameters():
        param.requires_grad = True


def unfreeze_all(bundle: ModelBundle) -> None:
    """Unfreeze every parameter in the model."""
    for param in bundle.model.parameters():
        param.requires_grad = True
