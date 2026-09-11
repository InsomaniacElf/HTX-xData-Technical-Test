"""Load the required official model with a narrow legacy-checkpoint allowlist."""
import os


def load_embedding_model():
    import torch
    from huggingface_hub import hf_hub_download
    from pyannote.audio import Model
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    from pyannote.audio.core.model import Introspection
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import Metadata, ContainerMetadata
    from omegaconf.nodes import AnyNode
    from collections import defaultdict
    from typing import Any
    from torch.torch_version import TorchVersion

    checkpoint = hf_hub_download("pyannote/embedding", "pytorch_model.bin",
        revision="4db4899737a38b2d618bbd74350915aa10293cb2", token=os.environ.get("HF_TOKEN"))
    allowed = [EarlyStopping, ModelCheckpoint, Specifications, Problem, Resolution,
               TorchVersion, Introspection, DictConfig, ListConfig, Metadata,
               ContainerMetadata, AnyNode, defaultdict, Any, int, list, dict]
    with torch.serialization.safe_globals(allowed):
        model = Model.from_pretrained(checkpoint)
    if model is None:
        raise RuntimeError("Unable to load pyannote/embedding")
    return model, checkpoint
