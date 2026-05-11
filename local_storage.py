# local_storage.py
from pathlib import Path
import pandas as pd
import shutil

class LocalStorage:
    """
    Local filesystem-based storage backend.
    This replaces OneDrive / SharePoint when Azure AD access is unavailable.
    """

    BASE = Path("data_storage")

    RAW = BASE / "raw"
    VALIDATED = BASE / "validated"
    CONSOLIDATED = BASE / "consolidated"
    REPORTS = BASE / "reports"

    def __init__(self):
        # Auto-create folder structure
        for path in [self.RAW, self.VALIDATED, self.CONSOLIDATED, self.REPORTS]:
            path.mkdir(parents=True, exist_ok=True)

    # ---------- GENERIC FILE OPS ----------

    def list_files(self, folder: Path):
        return list(folder.glob("*"))

    def save_file(self, src_path: str, dest_folder: Path, dest_name: str | None = None):
        src = Path(src_path)
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / (dest_name or src.name)
        shutil.copy(src, dest)
        return dest

    def load_file(self, folder: Path, filename: str) -> Path:
        file_path = folder / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found in {folder}")
        return file_path

    # ---------- BENCHMARK DATA ----------

    def load_benchmark_data(self) -> pd.DataFrame:
        """
        Load consolidated benchmarking dataset.
        Expected format: CSV stored in data_storage/consolidated/
        """
        files = list(self.CONSOLIDATED.glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                "No consolidated benchmark CSV found in data_storage/consolidated/"
            )
        latest = sorted(files)[-1]
        return pd.read_csv(latest)

    def save_benchmark_data(self, df: pd.DataFrame, filename="consolidated_benchmark.csv"):
        path = self.CONSOLIDATED / filename
        df.to_csv(path, index=False)
        return path


def get_storage():
    """
    Factory — future-proof.
    Later this can be swapped with SharePointStorage without touching app.py
    """
    return LocalStorage()