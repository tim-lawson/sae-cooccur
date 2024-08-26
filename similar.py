import sqlite3
from dataclasses import dataclass
from os import PathLike

import pandas as pd
import torch
from sae import Sae
from simple_parsing import Serializable, parse


@dataclass
class Config(Serializable):
    database: str = "cooccur.db"


def create_db(database: str | PathLike) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cossim (
        latent_i INTEGER,
        latent_j INTEGER,
        cossim INTEGER,
        PRIMARY KEY (latent_i, latent_j)
    )
    """)
    conn.commit()
    return conn, cursor


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, 1e-8 * torch.ones_like(norm))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda"
    config = parse(Config)

    conn, cursor = create_db(config.database)
    cursor.execute("DELETE FROM cossim")

    layers = ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5"]
    saes = Sae.load_many("EleutherAI/sae-pythia-70m-deduped-32k", layers=layers)
    saes = {layer: sae.to(device) for layer, sae in saes.items()}
    W_decs = [normalize(sae.W_dec, 1) for sae in saes.values() if sae.W_dec is not None]
    d_sae = W_decs[0].shape[0]

    cursor.execute("SELECT * FROM ppmi")
    for latent_i, latent_j, ppmi in cursor.fetchall():
        layer_i = latent_i // d_sae
        layer_j = latent_j // d_sae
        cossim = torch.dot(
            W_decs[layer_i][latent_i % d_sae], W_decs[layer_j][latent_j % d_sae]
        )
        cursor.execute(
            "INSERT INTO cossim (latent_i, latent_j, cossim) VALUES (?, ?, ?)",
            (latent_i, latent_j, cossim.item()),
        )
    conn.commit()

    cursor.execute("SELECT * FROM cossim")
    cossim = cursor.fetchall()
    cossim = pd.DataFrame(cossim, columns=["latent_i", "latent_j", "cossim"])

    cursor.execute("SELECT * FROM ppmi")
    ppmi = cursor.fetchall()
    ppmi = pd.DataFrame(ppmi, columns=["latent_i", "latent_j", "ppmi"])

    df = cossim.merge(ppmi, on=["latent_i", "latent_j"])
    df.to_csv("cossim_ppmi.csv", index=False)
