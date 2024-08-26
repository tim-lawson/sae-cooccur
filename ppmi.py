import torch
from safetensors.torch import load_file, save_file

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda"
    cooccur = load_file("cooccur.safetensors", device=device)
    cooccur = torch.sparse_coo_tensor(
        cooccur["indices"],
        cooccur["values"],
        size=torch.Size(cooccur["size"].tolist()),
        device=device,
    )
    count = cooccur.sum()
    count_i = cooccur.sum(dim=0).to_dense()
    count_j = cooccur.sum(dim=1).to_dense()
    cooccur = cooccur.coalesce()
    indices = cooccur.indices()
    ppmi = torch.sparse_coo_tensor(
        indices,
        torch.log2(
            count * cooccur.values() / count_i[indices[0]] / count_j[indices[1]]
        ),
        size=cooccur.size(),
        device=device,
    ).coalesce()
    save_file(
        {
            "indices": ppmi.indices().contiguous(),
            "values": ppmi.values().contiguous(),
            "size": torch.tensor(ppmi.size()),
        },
        "ppmi.safetensors",
    )
