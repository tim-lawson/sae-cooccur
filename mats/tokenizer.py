import math

from datasets.formatting.formatting import LazyBatch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/data.py
def concat_and_tokenize(
    batch: LazyBatch, tokenizer: PreTrainedTokenizerBase, max_length: int, key: str
) -> dict:
    output = tokenizer(
        tokenizer.eos_token.join([""] + batch[key]),  # type: ignore
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_overflowing_tokens=True,
    )
    overflowing_tokens = output.pop("overflowing_tokens", None)
    _ = output.pop("overflow_to_sample_mapping", None)
    if overflowing_tokens is not None:
        output["input_ids"] += [
            overflowing_tokens[i * max_length : (i + 1) * max_length]
            for i in range(math.ceil(len(overflowing_tokens) / max_length))
        ]  # type: ignore
    return {k: v[:-1] for k, v in output.items()}
