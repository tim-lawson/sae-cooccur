from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from simple_parsing import Serializable, parse
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from sae_cooccur.data import dataloader, load_data
from sae_cooccur.sae import load_saes_pythia
from sae_cooccur.tokenizer import load_tokenizer
from sae_cooccur.transformer import load_transformer


@dataclass
class Config(Serializable):
    batch_size: int = 1
    n_batches: int = 512  # 512 * 2048 ~ 1M tokens


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    model_name = "EleutherAI/pythia-70m-deduped"
    saes, layers = load_saes_pythia(device)
    transformer = load_transformer(model_name)
    tokenizer = load_tokenizer(model_name)
    dataset = load_data(
        tokenizer,
        "fahamu/ioi",
        "ioi_sentences",
        transformer.config.max_position_embeddings,
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

        for (layer, sae), hidden_state in zip(
            saes.items(), hidden_states, strict=False
        ):
            # batch pos k
            top_indices = sae.encode(hidden_state).top_indices
            # batch pos k -> (batch pos) k
            latents[layer].append(top_indices.view(-1, top_indices.size(-1)))

    latents = {layer: torch.cat(latent) for layer, latent in latents.items()}
    save_file(latents, "latents.safetensors")
