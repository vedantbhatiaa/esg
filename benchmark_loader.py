from pathlib import Path
import pandas as pd
from local_storage import LocalStorage

DATA_FILE = "data_storage/consolidated/consolidated_benchmarking.csv"


class BenchmarkLoader:
    def __init__(self, data_dir: Path | str = "."):
        self.data_dir = Path(data_dir)
        self.storage = LocalStorage()

    def load_raw(self) -> pd.DataFrame:
        """
        Load raw benchmarking CSV.
        """
        file_path = self.data_dir / DATA_FILE
        return pd.read_csv(file_path)

    def load_clean(self) -> pd.DataFrame:
        """
        Load and clean benchmarking data:
        - Drop empty rows
        - Convert numeric strings
        - Standardise column names
        """
        df = self.load_raw()

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Drop fully empty rows
        df = df.dropna(how="all")

        # Clean numeric data column
        if "data" in df.columns:
            df["data"] = (
                df["data"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df["data"] = pd.to_numeric(df["data"], errors="coerce")

        return df

    def filter_metric(
        self,
        section: str,
        row_label: str | None = None
    ) -> pd.DataFrame:
        """
        Convenience filter for a specific KPI or metric.
        """
        df = self.load_clean()

        mask = df["section"].str.lower() == section.lower()
        if row_label:
            mask &= df["row_label"].str.lower() == row_label.lower()

        return df.loc[mask]