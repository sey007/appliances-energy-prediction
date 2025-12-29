# The goal of this file is to load our chosen dataset.


import pandas as pd
from pathlib import Path


def load_energy_data(data_dir: Path) -> pd.DataFrame:
    
    csv_path = data_dir / "appliances_energy.csv"
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    df = load_energy_data(data_dir)
