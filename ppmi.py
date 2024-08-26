import sqlite3
from dataclasses import dataclass

import torch
from simple_parsing import Serializable, parse


@dataclass
class Config(Serializable):
    database: str = "cooccur.db"


def load_db(db_path) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ppmi (
        latent_i INTEGER,
        latent_j INTEGER,
        ppmi REAL,
        PRIMARY KEY (latent_i, latent_j)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS count_i (
        latent_i INTEGER,
        count INTEGER,
        PRIMARY KEY (latent_i)
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS count_j (
        latent_j INTEGER,
        count INTEGER,
        PRIMARY KEY (latent_j)
    )
    """)
    conn.commit()
    return conn, cursor


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)

    conn, cursor = load_db(config.database)
    cursor.execute("DELETE FROM count_i")
    cursor.execute("DELETE FROM count_j")
    cursor.execute("DELETE FROM ppmi")

    cursor.execute("""
    INSERT INTO count_i (latent_i, count)
    SELECT latent_i, SUM(count) FROM counts GROUP BY latent_i
    """)
    cursor.execute("""
    INSERT INTO count_j (latent_j, count)
    SELECT latent_j, SUM(count) FROM counts GROUP BY latent_j
    """)

    cursor.execute("SELECT SUM(count) FROM counts")
    count: int = cursor.fetchone()[0]

    cursor.execute(
        """
    INSERT INTO ppmi (latent_i, latent_j, ppmi)
    SELECT counts.latent_i,
    counts.latent_j,
    LOG(2, ? * counts.count / (count_i.count * count_j.count))
    FROM counts
    JOIN count_i ON counts.latent_i = count_i.latent_i
    JOIN count_j ON counts.latent_j = count_j.latent_j
    WHERE counts.count > 0
    AND count_i.count > 0
    AND count_j.count > 0
    """,
        (count,),
    )

    cursor.execute("DELETE FROM ppmi WHERE ppmi <= 0 OR ppmi IS NULL")
    conn.commit()
