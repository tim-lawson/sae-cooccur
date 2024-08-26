from multiprocessing import cpu_count

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from mats.tokenizer import concat_and_tokenize


def load_data(
    tokenizer: PreTrainedTokenizerBase, path: str, key: str, max_length: int
) -> IterableDataset:
    dataset: IterableDataset = load_dataset(
        path,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )  # type: ignore
    return dataset.map(
        concat_and_tokenize,
        batched=True,
        batch_size=1024,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "key": key,
        },
    ).with_format("torch")


def dataloader(
    dataset: IterableDataset, batch_size: int, num_workers: int = cpu_count() // 2
):
    return DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        num_workers=min(dataset.n_shards, num_workers),
    )
