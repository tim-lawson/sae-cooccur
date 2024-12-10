from collections.abc import Mapping
from typing import Literal

import torch
from sae import Sae
from sae_lens import SAE

type Saes = Mapping[str, torch.nn.Module]


def load_saes(
    SaeType: Literal["pythia", "gpt2"], device: torch.device | str
) -> tuple[Saes, list[str]]:
    if SaeType == "pythia":
        return load_saes_pythia(device)
    elif SaeType == "gpt2":
        return load_saes_gpt2(device)


def load_saes_pythia(device: torch.device | str):
    layers = [f"layers.{layer}" for layer in range(6)]
    saes = Sae.load_many("EleutherAI/sae-pythia-70m-deduped-32k", layers=layers)
    return {layer: sae.to(device) for layer, sae in saes.items()}, layers


def load_saes_gpt2(device: torch.device | str):
    layers = list(range(12))
    saes = {
        f"layers.{layer}": SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=str(device),
        )[0]
        for layer in layers
    }
    return saes, list(saes.keys())


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, 1e-8 * torch.ones_like(norm))
