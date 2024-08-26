import torch
from sae import Sae
from sae_lens import SAE


def load_pythia(device: torch.device | str):
    layers = [f"layers.{layer}" for layer in range(6)]
    saes = Sae.load_many("EleutherAI/sae-pythia-70m-deduped-32k", layers=layers)
    return {layer: sae.to(device) for layer, sae in saes.items()}, layers


def load_gpt2(device: torch.device | str):
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
