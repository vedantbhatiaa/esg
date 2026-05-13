"""
local_storage.py  —  TIP ESG Platform
=======================================
Local filesystem storage backend with the new organised folder structure.

data_storage/
├── master/      ← master CSV/Excel files (built by build_esg_master.py)
├── members/
│   ├── TIP/     ← per-TIP-company data folders
│   └── non_TIP/ ← future non-TIP companies
├── versions/    ← Parquet audit trail (one file per save event)
└── reports/
    ├── TIP/     ← generated reports per TIP company
    └── non_TIP/
"""
from pathlib import Path
import pandas as pd
import shutil


class LocalStorage:
    BASE        = Path("data_storage")
    MASTER      = BASE / "master"
    MEMBERS_TIP = BASE / "members" / "TIP"
    MEMBERS_NON = BASE / "members" / "non_TIP"
    VERSIONS    = BASE / "versions"
    REPORTS_TIP = BASE / "reports" / "TIP"
    REPORTS_NON = BASE / "reports" / "non_TIP"

    def __init__(self):
        for path in [
            self.MASTER, self.MEMBERS_TIP, self.MEMBERS_NON,
            self.VERSIONS, self.REPORTS_TIP, self.REPORTS_NON,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # ── Generic ───────────────────────────────────────────────────────────────

    def list_files(self, folder: Path):
        return list(folder.glob("*")) if folder.exists() else []

    def save_file(self, src_path: str, dest_folder: Path, dest_name: str = None):
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

    # ── Master data ───────────────────────────────────────────────────────────

    def load_master(self) -> pd.DataFrame:
        """Load the wide master CSV from data_storage/master/."""
        candidates = [
            self.MASTER / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv",
        ]
        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError(
            "Master CSV not found. Run python build_esg_master.py first."
        )

    def save_master(self, df: pd.DataFrame):
        path = self.MASTER / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"
        df.to_csv(path, index=False)
        return path

    # ── Member company files ───────────────────────────────────────────────────

    def get_member_folder(self, company: str, tip: bool = True) -> Path:
        base = self.MEMBERS_TIP if tip else self.MEMBERS_NON
        folder = base / company.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_member_data(self, company: str, df: pd.DataFrame, tip: bool = True):
        folder   = self.get_member_folder(company, tip)
        filename = f"{company.replace(' ','_')}_latest.csv"
        path     = folder / filename
        df.to_csv(path, index=False)
        return path

    # ── Versions (Parquet audit trail) ────────────────────────────────────────

    def save_version(self, company: str, year: int, df: pd.DataFrame) -> str:
        """
        Save a Parquet snapshot to data_storage/versions/.
        Filename: CompanyName_Year_YYYYMMDD_HHMMSS.parquet
        These are NEVER overwritten — each save event creates a new file.
        """
        from datetime import datetime
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        co_safe   = company.replace(" ", "_").replace("/", "_")
        filename  = f"{co_safe}_{year}_{ts}.parquet"
        out_path  = self.VERSIONS / filename
        df.to_parquet(out_path, index=False)
        return filename

    def list_versions(self, company: str = None, year: int = None) -> list:
        """List all version files, optionally filtered by company and/or year."""
        all_files = sorted(self.VERSIONS.glob("*.parquet"), reverse=True)
        if company:
            co_safe   = company.replace(" ", "_").replace("/", "_")
            all_files = [f for f in all_files if f.stem.startswith(co_safe)]
        if year:
            all_files = [f for f in all_files if f"_{year}_" in f.stem]
        return all_files

    def load_version(self, filename: str) -> pd.DataFrame:
        return pd.read_parquet(self.VERSIONS / filename)

    # ── Reports ───────────────────────────────────────────────────────────────

    def get_reports_folder(self, company: str, tip: bool = True) -> Path:
        base   = self.REPORTS_TIP if tip else self.REPORTS_NON
        folder = base / company.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_report(self, company: str, filename: str,
                    content: bytes, tip: bool = True) -> Path:
        folder = self.get_reports_folder(company, tip)
        path   = folder / filename
        path.write_bytes(content)
        return path

    # ── Benchmark data (legacy compatibility) ─────────────────────────────────

    def load_benchmark_data(self) -> pd.DataFrame:
        """Load consolidated benchmarking dataset from master/."""
        csv_files = list(self.MASTER.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                "No CSV found in data_storage/master/. "
                "Run python build_esg_master.py first."
            )
        latest = sorted(csv_files)[-1]
        return pd.read_csv(latest)


def get_storage() -> LocalStorage:
    """Factory — returns a LocalStorage instance."""
    return LocalStorage()