import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

if __name__ == "__main__":
    d_sae = 32768
    latents = load_file("latents.safetensors")
    for layer, indices in latents.items():
        n_tokens, k = indices.shape
    latents = [indices for indices in latents.values()]
    n_layers = len(latents)
    n_latents = n_layers * d_sae

    cooccur = torch.sparse_coo_tensor(torch.empty([2, 0]), [], [n_latents, n_latents])
    for token in tqdm(range(n_tokens)):
        indices = torch.cat(
            [indices[token] + d_sae * layer for layer, indices in enumerate(latents)]
        )
        pairs = torch.stack(
            [
                indices.unsqueeze(1).expand(-1, indices.size(0)).reshape(-1),
                indices.unsqueeze(0).expand(indices.size(0), -1).reshape(-1),
            ]
        )
        update = torch.sparse_coo_tensor(
            pairs, torch.ones(pairs.size(1)), [n_latents, n_latents]
        )
        torch.add(cooccur, update, out=cooccur)
    cooccur = cooccur.coalesce()
    save_file(
        {
            "indices": cooccur.indices().contiguous(),
            "values": cooccur.values().contiguous(),
            "size": torch.tensor(cooccur.size()),
        },
        "cooccur.safetensors",
    )
