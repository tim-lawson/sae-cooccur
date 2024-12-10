import os
from dataclasses import dataclass
from itertools import product
from typing import Literal

import pandas as pd
import torch
from simple_parsing import Serializable, parse

from sae_cooccur.sae import load_saes, normalize


@dataclass
class Config(Serializable):
    saes: Literal["pythia", "gpt2"]
    bins: int = 201


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)
    os.makedirs("out", exist_ok=True)

    saes, layers = load_saes(config.saes, device)
    W_decs = [normalize(sae.W_dec, 1) for sae in saes.values()]
    d_sae, d_model = W_decs[0].shape

    mmcs, hist, bins = [], {}, torch.linspace(-1, 1, config.bins).cpu().numpy()
    for i, j in product(range(len(W_decs)), repeat=2):
        cos_sim = torch.mm(W_decs[i], W_decs[j].T)
        cos_sim = torch.triu(cos_sim, diagonal=1)
        cos_sim_max = cos_sim.max(dim=0).values
        mmcs.append(
            {
                "layer_i": i,
                "layer_j": j,
                "mean": cos_sim_max.mean().item(),
                "var": cos_sim_max.var().item(),
                "std": cos_sim_max.std().item(),
                "min": cos_sim_max.min().item(),
                "max": cos_sim_max.max().item(),
            }
        )
        cos_sim = cos_sim.flatten()
        hist[f"layer_{i}_{j}"] = (
            torch.histc(cos_sim, bins=config.bins, min=-1, max=+1).cpu().numpy()
        )
    pd.DataFrame({"bins": bins, **hist}).to_csv(
        f"out/cshist_{config.saes}.csv", index=False
    )
    pd.DataFrame(mmcs).to_csv(f"out/mmcs_{config.saes}.csv", index=False)
