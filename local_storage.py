"""
local_storage.py — TIP ESG Platform
=======================================
Local filesystem storage backend with the organised folder structure.

data_storage/
├── master/          ← master CSV/Excel files
├── members/
│   ├── TIP/         ← per-TIP-company data folders
│   └── non_TIP/     ← future non-TIP companies
├── versions/
│   └── {CompanyName}/   ← ONE subfolder per company (M5 FIX)
│       └── CompanyName_Year_YYYYMMDD_HHMMSS.parquet
└── reports/
    ├── TIP/
    └── non_TIP/

v1 hardening changes:
- M5: save_version() now creates a per-company subfolder inside versions/
      matching the structure used by app.py's _save_version_parquet().
      Previously flat saves and subfolder saves produced diverging layouts.
- L3: save_member_data() now keeps a versioned copy (timestamp-suffixed)
      alongside the _latest.csv file so prior member snapshots are never
      silently overwritten.
"""

from pathlib import Path
from datetime import datetime
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

    def save_file(self, src_path: str, dest_folder: Path,
                  dest_name: str = None):
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
        candidate = self.MASTER / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"
        if candidate.exists():
            return pd.read_csv(candidate)
        raise FileNotFoundError(
            "Master CSV not found. Run python build_esg_master.py first."
        )

    def save_master(self, df: pd.DataFrame) -> Path:
        path = self.MASTER / "ESG_MASTER_WIDE_ALL_COMPANIES_2009_2023.csv"
        df.to_csv(path, index=False)
        return path

    # ── Member company files ───────────────────────────────────────────────────

    def get_member_folder(self, company: str, tip: bool = True) -> Path:
        base   = self.MEMBERS_TIP if tip else self.MEMBERS_NON
        folder = base / company.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_member_data(self, company: str, df: pd.DataFrame,
                         tip: bool = True) -> Path:
        """
        Save member company data.

        L3 FIX: In addition to overwriting _latest.csv (for easy access to
        the current snapshot), a timestamped copy is written to the same
        folder so no prior submission is ever silently destroyed.
        """
        folder    = self.get_member_folder(company, tip)
        co_safe   = company.replace(" ", "_")
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

        latest_path  = folder / f"{co_safe}_latest.csv"
        version_path = folder / f"{co_safe}_{ts}.csv"

        df.to_csv(latest_path,  index=False)   # always up-to-date pointer
        df.to_csv(version_path, index=False)   # L3 FIX — immutable snapshot

        return latest_path

    # ── Versions (Parquet audit trail) ────────────────────────────────────────

    def save_version(self, company: str, year: int,
                     df: pd.DataFrame) -> str:
        """
        Save a Parquet snapshot.

        M5 FIX: Files are now stored in a per-company subfolder
        (data_storage/versions/{CompanyName}/) to match the structure used by
        app.py's _save_version_parquet().  Previously flat saves to
        data_storage/versions/ diverged from the app.py layout.

        Files are NEVER overwritten — each save event creates a new file.
        """
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        co_safe = company.replace(" ", "_").replace("/", "_")
        fname   = f"{co_safe}_{year}_{ts}.parquet"

        # M5 FIX — subfolder per company
        ver_dir = self.VERSIONS / co_safe
        ver_dir.mkdir(parents=True, exist_ok=True)

        out_path = ver_dir / fname
        df.to_parquet(out_path, index=False)
        return f"{co_safe}/{fname}"

    def list_versions(self, company: str = None, year: int = None) -> list:
        """List all version files, optionally filtered by company and/or year.
        M5 FIX: searches inside per-company subfolders.
        """
        if company:
            co_safe = company.replace(" ", "_").replace("/", "_")
            search_root = self.VERSIONS / co_safe
        else:
            search_root = self.VERSIONS

        all_files = sorted(search_root.rglob("*.parquet"), reverse=True)

        if year:
            all_files = [f for f in all_files if f"_{year}_" in f.stem]

        return all_files

    def load_version(self, filename: str) -> pd.DataFrame:
        """Load a version Parquet. Accepts either a bare filename or a
        company/filename relative path."""
        p = self.VERSIONS / filename
        if not p.exists():
            raise FileNotFoundError(f"Version not found: {filename}")
        return pd.read_parquet(p)

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

    # ── Benchmark data ────────────────────────────────────────────────────────

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