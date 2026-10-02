"""Shared OpenCLIP loading helpers for detector and metric evaluation."""

from __future__ import annotations


def load_open_clip(model_name, *, pretrained="openai", device="cuda", cache_dir=None,
                   precision=None):
    """Load an OpenCLIP model, preprocessing transform, and matching tokenizer.

    ``pretrained`` may be an OpenCLIP tag (for example ``openai``) or a local
    checkpoint path. Keeping architecture and weights separate lets compatible
    metrics reuse the same code while swapping only weights.
    """
    import open_clip

    kwargs = {"pretrained": pretrained, "device": device}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if precision is not None:
        kwargs["precision"] = precision
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, **kwargs)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(model_name)
