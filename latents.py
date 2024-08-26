from dataclasses import dataclass

import torch
from sae import Sae
from safetensors.torch import save_file
from simple_parsing import Serializable, field, parse
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from mats.data import dataloader, load_data
from mats.tokenizer import load_tokenizer
from mats.transformer import load_transformer


@dataclass
class Config(Serializable):
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layers: list[str] = field(
        default_factory=lambda: [
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "layers.4",
            "layers.5",
        ]
    )
    batch_size: int = 1
    n_batches: int = 512


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    config = parse(Config)

    transformer = load_transformer(config.model_name)
    tokenizer = load_tokenizer(config.model_name)
    dataset = load_data(tokenizer, transformer.config.max_position_embeddings)
    saes = Sae.load_many("EleutherAI/sae-pythia-70m-deduped-32k", layers=config.layers)
    saes = {layer: sae.to(transformer.device) for layer, sae in saes.items()}

    latents = {layer: [] for layer in config.layers}
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
            top_indices = sae.encode(hidden_state).top_indices
            # batch pos k -> (batch pos) k
            latents[layer].append(top_indices.view(-1, top_indices.size(-1)))

    latents = {layer: torch.cat(latent) for layer, latent in latents.items()}
    save_file(latents, "latents.safetensors")
