import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike

import torch
from safetensors.torch import load_file
from simple_parsing import Serializable, parse
from tqdm import tqdm


@dataclass
class Config(Serializable):
    # 32768 for EleutherAI/pythia-70m-deduped
    # 24576 for openai-community/gpt2
    d_sae: int
    batch_size: int = 1_000_000
    latents: str = "latents.safetensors"
    database: str = "cooccur.db"


def create_db(database: str | PathLike) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        latent_i INTEGER,
        latent_j INTEGER,
        count INTEGER,
        PRIMARY KEY (latent_i, latent_j)
    )
    """)
    conn.commit()
    return conn, cursor


def update(cursor: sqlite3.Cursor, batch) -> None:
    cursor.executemany(
        """
    INSERT INTO counts (latent_i, latent_j, count)
    VALUES (?, ?, ?)
    ON CONFLICT(latent_i, latent_j) DO UPDATE SET count = count + excluded.count
    """,
        [(i, j, count) for (i, j), count in batch.items()],
    )


def find_pairs(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.combinations(indices, r=2)
    pairs = torch.min(pairs, pairs.flip(1))
    return torch.unique(pairs, dim=0, return_counts=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    latents = load_file(config.latents, device=device)
    for layer, indices in latents.items():
        n_tokens, k = indices.shape
    latents = [indices for indices in latents.values()]
    n_layers = len(latents)
    n_latents = n_layers * config.d_sae

    conn, cursor = create_db(config.database)
    batch = defaultdict(int)
    for token in tqdm(range(n_tokens)):
        pairs, counts = find_pairs(
            torch.cat(
                [
                    indices[token] + config.d_sae * layer
                    for layer, indices in enumerate(latents)
                ]
            )
        )
        for (i, j), count in zip(pairs.tolist(), counts.tolist()):
            batch[(i, j)] += count
        if len(batch) >= config.batch_size:
            update(cursor, batch)
            conn.commit()
            batch.clear()
    if batch:
        update(cursor, batch)
        conn.commit()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_count ON counts(count)")
    conn.commit()
    conn.close()
