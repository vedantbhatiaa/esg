"""
TIP ESG Platform — Storage Module
====================================
Uses Microsoft Graph API to store / retrieve files from OneDrive / SharePoint.

Why OneDrive / SharePoint (not AWS S3 or Azure Blob):
  • dss+ is on Microsoft 365 E5 → 1 TB OneDrive per user already licensed
  • All files stay inside the existing Microsoft 365 tenant (no new cloud account)
  • SharePoint = multi-user collaborative access with live sync
  • E5 Purview DLP governs what leaves the tenant automatically
  • Microsoft's contractual commitment: zero data used for training, zero human access

Setup (one-time — IT or manager):
  1. Register an Azure AD App in portal.azure.com
  2. Grant: Files.ReadWrite.All + Sites.ReadWrite.All (application permissions)
  3. Copy CLIENT_ID, CLIENT_SECRET, TENANT_ID into your .env file

Privacy note:
  Raw company ESG files are stored in a SharePoint document library with:
  - Role-based permissions (only dss+ analysts + named client contacts)
  - Azure Purview sensitivity label: "Confidential – Client ESG Data"
  - Retention policy: 7 years (configurable)
"""

import os, io, json, logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

import requests

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
CLIENT_ID     = os.getenv("AZURE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
TENANT_ID     = os.getenv("AZURE_TENANT_ID", "")

# SharePoint site URL — the TIP ESG project site
SHAREPOINT_SITE   = os.getenv("SHAREPOINT_SITE", "consultdss.sharepoint.com:/sites/TIP-ESG")
SHAREPOINT_DRIVE  = os.getenv("SHAREPOINT_DRIVE", "TIP-ESG-Data")   # document library name
GRAPH_BASE        = "https://graph.microsoft.com/v1.0"

# Folder structure inside the document library
FOLDER_RAW_TEMPLATES  = "01_Templates_Raw"      # incoming company submissions
FOLDER_VALIDATED      = "02_Validated"           # after dss+ review
FOLDER_CONSOLIDATED   = "03_Consolidated"        # master consolidated workbook
FOLDER_REPORTS        = "04_Reports"             # generated populated reports
FOLDER_ARCHIVE        = "99_Archive"             # superseded versions


class StorageClient:
    """
    Handles all file operations with SharePoint / OneDrive via Microsoft Graph.
    
    Authentication: App-only (service principal) — no user credentials stored.
    Token is fetched on first call and cached in memory. Refreshed automatically.
    """

    def __init__(self):
        self._token: Optional[str]  = None
        self._token_expiry: float   = 0
        self._drive_id: Optional[str] = None
        self._site_id: Optional[str]  = None

    # ── Authentication ────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """Acquire an OAuth2 access token using client_credentials flow."""
        import time
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        data = {
            "grant_type":    "client_credentials",
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope":         "https://graph.microsoft.com/.default",
        }
        resp = requests.post(url, data=data, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        import time as _t
        self._token        = body["access_token"]
        self._token_expiry = _t.time() + body.get("expires_in", 3600)
        logger.info("Access token acquired (expires in %ss)", body.get("expires_in"))
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}",
                "Content-Type":  "application/json"}

    def _get_site_id(self) -> str:
        """Resolve SharePoint site path to a site_id (cached)."""
        if self._site_id:
            return self._site_id
        url  = f"{GRAPH_BASE}/sites/{SHAREPOINT_SITE}"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        self._site_id = resp.json()["id"]
        return self._site_id

    def _get_drive_id(self) -> str:
        """Resolve document library name to drive_id (cached)."""
        if self._drive_id:
            return self._drive_id
        site_id = self._get_site_id()
        url     = f"{GRAPH_BASE}/sites/{site_id}/drives"
        resp    = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        drives = resp.json().get("value", [])
        for d in drives:
            if d.get("name") == SHAREPOINT_DRIVE:
                self._drive_id = d["id"]
                return self._drive_id
        raise ValueError(f"Document library '{SHAREPOINT_DRIVE}' not found in site.")

    # ── Core operations ───────────────────────────────────────────────────────

    def upload(self, local_path: Union[str, Path], remote_folder: str,
               remote_filename: Optional[str] = None) -> dict:
        """
        Upload a file to SharePoint.

        Args:
            local_path:      Path to the local file (e.g. "TEMPLATE_VerdaTyres_2021.xlsx")
            remote_folder:   SharePoint folder (e.g. "01_Templates_Raw/Bridgestone")
            remote_filename: Override filename (defaults to local filename)

        Returns:
            Graph API response dict with id, webUrl, etc.
        """
        local_path  = Path(local_path)
        fname       = remote_filename or local_path.name
        drive_id    = self._get_drive_id()
        remote_path = f"{remote_folder}/{fname}".lstrip("/")
        url         = f"{GRAPH_BASE}/drives/{drive_id}/root:/{remote_path}:/content"

        with open(local_path, "rb") as fh:
            content = fh.read()

        headers = {**self._headers(),
                   "Content-Type": "application/octet-stream"}
        del headers["Content-Type"]   # Graph sets it from the binary stream
        headers["Authorization"] = f"Bearer {self._get_token()}"
        headers["Content-Type"]  = "application/octet-stream"

        resp = requests.put(url, headers=headers, data=content, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        logger.info("Uploaded %s → %s (id=%s)", fname, remote_path, result.get("id"))
        return result

    def download(self, remote_folder: str, remote_filename: str,
                 local_path: Optional[Union[str, Path]] = None) -> bytes:
        """
        Download a file from SharePoint.

        Returns raw bytes. Optionally saves to local_path.
        """
        drive_id    = self._get_drive_id()
        remote_path = f"{remote_folder}/{remote_filename}".lstrip("/")
        url         = f"{GRAPH_BASE}/drives/{drive_id}/root:/{remote_path}:/content"

        resp = requests.get(url, headers=self._headers(), timeout=60, allow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        if local_path:
            Path(local_path).write_bytes(data)
            logger.info("Downloaded %s → %s", remote_path, local_path)
        return data

    def list_files(self, remote_folder: str) -> list[dict]:
        """List all files in a SharePoint folder."""
        drive_id = self._get_drive_id()
        url      = f"{GRAPH_BASE}/drives/{drive_id}/root:/{remote_folder}:/children"
        resp     = requests.get(url, headers=self._headers(), timeout=15)
        resp.raise_for_status()
        items = resp.json().get("value", [])
        return [{"name": i["name"], "size": i.get("size"),
                 "modified": i.get("lastModifiedDateTime"),
                 "url": i.get("webUrl"), "id": i["id"]} for i in items
                if "folder" not in i]

    def archive(self, remote_folder: str, filename: str) -> None:
        """
        Move a file to the Archive folder (versioning instead of deletion).
        Keeps an audit trail — files are never permanently deleted.
        """
        drive_id = self._get_drive_id()
        src_path = f"{remote_folder}/{filename}"
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dst_path = f"{FOLDER_ARCHIVE}/{remote_folder.replace('/','_')}_{ts}_{filename}"

        # Get source item id
        url_src  = f"{GRAPH_BASE}/drives/{drive_id}/root:/{src_path}"
        src_item = requests.get(url_src, headers=self._headers()).json()
        item_id  = src_item["id"]

        # Move via PATCH
        payload = {"parentReference": {"path": f"/drives/{drive_id}/root:/{FOLDER_ARCHIVE}"}}
        url_mv  = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}"
        requests.patch(url_mv, headers=self._headers(), json=payload).raise_for_status()
        logger.info("Archived %s → %s", src_path, dst_path)

    def save_metadata(self, company: str, year: int, meta: dict) -> None:
        """
        Save submission metadata as a JSON sidecar file alongside the Excel.
        Enables audit trail and submission tracking without querying file contents.
        """
        meta["company"]    = company
        meta["year"]       = year
        meta["timestamp"]  = datetime.utcnow().isoformat()
        content = json.dumps(meta, indent=2).encode()

        drive_id    = self._get_drive_id()
        remote_path = f"{FOLDER_VALIDATED}/{company}/{year}_metadata.json"
        url         = f"{GRAPH_BASE}/drives/{drive_id}/root:/{remote_path}:/content"
        headers     = {"Authorization": f"Bearer {self._get_token()}",
                       "Content-Type": "application/octet-stream"}
        requests.put(url, headers=headers, data=content).raise_for_status()
        logger.info("Metadata saved for %s %s", company, year)


# ── Convenience helpers ────────────────────────────────────────────────────────

_client: Optional[StorageClient] = None

def get_client() -> StorageClient:
    global _client
    if _client is None:
        _client = StorageClient()
    return _client


def upload_template(file_path: str, company: str) -> dict:
    """Upload a newly received company template to the raw submissions folder."""
    return get_client().upload(file_path, f"{FOLDER_RAW_TEMPLATES}/{company}")


def upload_validated(file_path: str, company: str) -> dict:
    """Upload an analyst-approved template to the validated folder."""
    return get_client().upload(file_path, f"{FOLDER_VALIDATED}/{company}")


def upload_report(file_path: str, company: str, year: int) -> dict:
    """Upload a generated populated report."""
    fname = f"{company.replace(' ','_')}_{year}_ESG_Report.xlsx"
    return get_client().upload(file_path, f"{FOLDER_REPORTS}/{company}", fname)


def download_template(company: str, filename: str, save_to: str = ".") -> bytes:
    """Download the latest template for a company."""
    return get_client().download(f"{FOLDER_RAW_TEMPLATES}/{company}", filename, save_to)


def list_submissions(company: str = None) -> list[dict]:
    """List all files in a company's raw templates folder."""
    folder = f"{FOLDER_RAW_TEMPLATES}/{company}" if company else FOLDER_RAW_TEMPLATES
    return get_client().list_files(folder)


# ── OneDrive personal (mock / dev use) ────────────────────────────────────────

class MockStorage:
    """
    Local filesystem mock of StorageClient.
    Use during development when SharePoint credentials aren't available.
    Mirrors the same folder structure on disk.
    """
    BASE = Path("./mock_storage")

    def __init__(self):
        for folder in [FOLDER_RAW_TEMPLATES, FOLDER_VALIDATED,
                       FOLDER_CONSOLIDATED, FOLDER_REPORTS, FOLDER_ARCHIVE]:
            (self.BASE / folder).mkdir(parents=True, exist_ok=True)
        print(f"MockStorage initialised at {self.BASE.resolve()}")

    def upload(self, local_path, remote_folder, remote_filename=None):
        local_path = Path(local_path)
        dest = self.BASE / remote_folder / (remote_filename or local_path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local_path.read_bytes())
        print(f"[MOCK] Uploaded {local_path.name} → {dest.relative_to(self.BASE)}")
        return {"name": dest.name, "webUrl": str(dest)}

    def download(self, remote_folder, remote_filename, local_path=None):
        src = self.BASE / remote_folder / remote_filename
        data = src.read_bytes()
        if local_path:
            Path(local_path).write_bytes(data)
        return data

    def list_files(self, remote_folder):
        folder = self.BASE / remote_folder
        if not folder.exists():
            return []
        return [{"name": f.name, "size": f.stat().st_size,
                 "modified": str(f.stat().st_mtime)} for f in folder.iterdir()
                if f.is_file()]

    def save_metadata(self, company, year, meta):
        meta.update({"company": company, "year": year,
                     "timestamp": datetime.utcnow().isoformat()})
        dest = self.BASE / FOLDER_VALIDATED / company / f"{year}_metadata.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(meta, indent=2))


def get_storage(mock: bool = False):
    """
    Factory. Use mock=True locally; production uses the real SharePoint client.
    """
    if mock or not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
        return MockStorage()
    return get_client()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    storage = get_storage(mock=True)
    
    # Create a test file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"TIP ESG test file")
        tmp = f.name
    
    result = storage.upload(tmp, FOLDER_RAW_TEMPLATES + "/VerdaTyres", "test_upload.txt")
    print("Upload result:", result)
    
    files = storage.list_files(FOLDER_RAW_TEMPLATES)
    print("Files in raw templates:", files)
    
    storage.save_metadata("VerdaTyres Corp", 2023, {"status": "approved", "flags": 0})
    print("Metadata saved")
    print("\nStorage module self-test passed.")
