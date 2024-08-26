import math
from itertools import product

import pandas as pd
import torch
from sae import Sae
from safetensors.torch import load_file
from tqdm import tqdm


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, 1e-8 * torch.ones_like(norm))


def topk_cos_sim(W_dec: torch.Tensor, k: int = 8, chunk_size: int = 256):
    W_dec = normalize(W_dec, dim=1)
    d_sae, d_model = W_dec.shape
    values = torch.zeros((d_sae, k), device=device)
    indices = torch.zeros((d_sae, k), device=device, dtype=torch.long)
    for i in range(0, d_sae, chunk_size):
        chunk_W_dec = W_dec[i : i + chunk_size, :]  # chunk_size d_model
        cos_sim = torch.mm(chunk_W_dec, W_dec.T)  # chunk_size d_sae
        mask = torch.ones_like(cos_sim, dtype=torch.bool, device=device)
        mask = torch.triu(mask[: i + chunk_size, :], diagonal=1)
        cos_sim = cos_sim.masked_fill(~mask[: i + chunk_size, :], float("-inf"))
        topk = torch.topk(cos_sim, k, dim=1)
        values_ = torch.cat((values[i : i + chunk_size, :], topk.values), dim=1)
        indices_ = torch.cat((indices[i : i + chunk_size, :], topk.indices), dim=1)
        values_, sort = torch.sort(values_, dim=1, descending=True)
        indices_ = torch.gather(indices_, 1, sort)
        values[i : i + chunk_size, :] = values_[:, :k]  # chunk_size k
        indices[i : i + chunk_size, :] = indices_[:, :k]
    return values, indices


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda"
    ppmi = load_file("ppmi.safetensors", device=device)
    ppmi = torch.sparse_coo_tensor(
        ppmi["indices"],
        ppmi["values"],
        size=torch.Size(ppmi["size"].tolist()),
        device=device,
    )
    layers = ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5"]
    saes = Sae.load_many("EleutherAI/sae-pythia-70m-deduped-32k", layers=layers)
    saes = {layer: sae.to(device) for layer, sae in saes.items()}
    data = []
    for layer, sae in enumerate(saes.values()):
        assert sae.W_dec is not None
        d_sae, d_model = sae.W_dec.shape
        cos_sim, features = topk_cos_sim(sae.W_dec)  # d_sae k
        k = cos_sim.shape[1]
        for feature_i, topk_i in tqdm(product(range(d_sae), range(k)), total=d_sae * k):
            feature_j = features[feature_i][topk_i].item()
            pair_cos_sim = cos_sim[feature_i][topk_i].item()
            pair_ppmi = ppmi[
                feature_i + layer * d_sae,
                features[feature_i][topk_i] + layer * d_sae,
            ].item()
            if not math.isclose(pair_cos_sim, 1.0, abs_tol=1e-3) and pair_ppmi > 0:
                data.append((layer, feature_i, feature_j, pair_cos_sim, pair_ppmi))
    df = pd.DataFrame(data, columns=["layer", "i", "j", "cos_sim", "ppmi"])
    df.to_csv("similar.csv", index=False)
