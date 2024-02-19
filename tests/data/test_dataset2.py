import pyrootutils
import hydra
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def test_dataset(cfg):
    root_vas= cfg.env.PATH_DATA_V
    root = cfg.env.PATH_DATA

    conn = sqlite3.connect(root_vas)
    c = conn.cursor()
    # print all tables and their columns
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    for table in tables:
        c.execute(f"PRAGMA table_info({table[0]})")
        print(table[0], c.fetchall())
    # In table VAS_notch load PSD_notch
    c.execute("SELECT PSD FROM AFTER_VAS")
    psd = c.fetchone()[0]
    psd = np.frombuffer(psd, dtype=np.float32)
    plt.plot(psd)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    test_dataset()