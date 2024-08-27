from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from simple_parsing import Serializable, parse
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from mats.data import dataloader, load_data
from mats.sae import load_saes_gpt2
from mats.tokenizer import load_tokenizer
from mats.transformer import load_transformer


@dataclass
class Config(Serializable):
    batch_size: int = 1
    n_batches: int = 1024  # 1024 * 1024 ~ 1M tokens
    k: int = 16


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    model_name = "openai-community/gpt2"
    saes, layers = load_saes_gpt2(device)
    transformer = load_transformer(model_name)
    tokenizer = load_tokenizer(model_name)
    dataset = load_data(
        tokenizer,
        "Skylion007/openwebtext",
        "text",
        transformer.config.n_positions,
    )

    latents = {layer: [] for layer in layers}
    for i, batch in tqdm(
        enumerate(dataloader(dataset, config.batch_size)), total=config.n_batches
    ):
        if i >= config.n_batches:
            break

        outputs: CausalLMOutputWithPast = transformer.forward(
            batch["input_ids"].to(transformer.device),
            output_hidden_states=True,
            return_dict=True,
        )  # type: ignore
        # batch pos d_model
        hidden_states = outputs.hidden_states[1:]  # type: ignore

        for (layer, sae), hidden_state in zip(saes.items(), hidden_states):
            # batch pos k
            top_indices = sae.encode(hidden_state).topk(config.k).indices
            # batch pos k -> (batch pos) k
            latents[layer].append(top_indices.view(-1, top_indices.size(-1)))

    latents = {layer: torch.cat(latent) for layer, latent in latents.items()}
    save_file(latents, "latents.safetensors")
