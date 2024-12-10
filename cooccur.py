import sqlite3
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
    max_tokens: int | None = None
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


def update(cursor: sqlite3.Cursor, batch: torch.Tensor) -> None:
    batch = batch.coalesce()
    cursor.executemany(
        """
    INSERT INTO counts (latent_i, latent_j, count)
    VALUES (?, ?, ?)
    ON CONFLICT(latent_i, latent_j) DO UPDATE SET count = excluded.count + count
    """,
        [
            (index[0], index[1], value)
            for index, value in zip(
                batch.indices().tolist(), batch.values().tolist(), strict=False
            )
        ],
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    latents = load_file(config.latents, device=device)
    n_tokens, k = 0, 0
    for _layer, indices in latents.items():
        n_tokens, k = indices.shape

    latents = [indices for indices in latents.values()]
    n_layers = len(latents)
    n_latents = n_layers * config.d_sae
    print({"n_layers": n_layers, "n_latents": n_latents, "n_tokens": n_tokens, "k": k})

    conn, cursor = create_db(config.database)
    batch = torch.sparse_coo_tensor(
        torch.empty([2, 0]), [], size=[n_latents, n_latents], device=device
    )
    for token in tqdm(range(config.max_tokens or n_tokens)):
        indices_token = torch.cat(
            [
                indices[token] + config.d_sae * layer
                for layer, indices in enumerate(latents)
            ]
        )
        pairs = torch.combinations(indices_token, r=2, with_replacement=True)
        torch.add(
            batch,
            torch.sparse_coo_tensor(
                pairs.permute(1, 0),
                torch.ones(pairs.size(0)),
                size=[n_latents, n_latents],
                device=device,
            ),
            out=batch,
        )
        if batch._nnz() >= config.batch_size:
            update(cursor, batch)
            conn.commit()
            batch = torch.zeros_like(batch)
    if batch._nnz() > 0:
        update(cursor, batch)
        conn.commit()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_count ON counts(count)")
    conn.commit()
    conn.close()
