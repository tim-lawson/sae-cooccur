import sqlite3
from dataclasses import dataclass
from os import PathLike
from typing import Literal

import pandas as pd
import torch
from simple_parsing import Serializable, parse
from tqdm import tqdm

from sae_cooccur.sae import load_saes, normalize


@dataclass
class Config(Serializable):
    saes: Literal["pythia", "gpt2"]
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


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    conn, cursor = create_db(config.database)
    cursor.execute("DELETE FROM cossim")

    saes, layers = load_saes(config.saes, device)
    W_decs = [normalize(sae.W_dec, 1) for sae in saes.values() if sae.W_dec is not None]
    d_sae = W_decs[0].shape[0]

    cursor.execute("SELECT * FROM ppmi")
    for latent_i, latent_j, _ppmi in tqdm(cursor.fetchall()):
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

    # TODO: Join tables with SQL.
    cursor.execute("SELECT * FROM cossim")
    cossim = cursor.fetchall()
    cossim = pd.DataFrame(cossim, columns=["latent_i", "latent_j", "cossim"])

    cursor.execute("SELECT * FROM ppmi")
    ppmi = cursor.fetchall()
    ppmi = pd.DataFrame(ppmi, columns=["latent_i", "latent_j", "ppmi"])

    df = cossim.merge(ppmi, on=["latent_i", "latent_j"])
    df.to_csv("cossim_ppmi.csv", index=False)
