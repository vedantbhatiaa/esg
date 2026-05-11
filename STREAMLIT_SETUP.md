# TIP ESG Platform — Setup & Run Guide
## Fixing Streamlit in PowerShell + Full Project Structure

---

## 1  Why Streamlit Fails in PowerShell (and the Fix)

PowerShell blocks running `.exe` scripts installed by pip by default.
The `streamlit` command (a script in Python's `Scripts\` folder) gets blocked by the execution policy.

### Fix A — Use `python -m` instead (recommended, always works)
```powershell
# Instead of:
streamlit run app.py          # ← this fails

# Use this:
python -m streamlit run app.py    # ← this always works
```

### Fix B — Fix the execution policy (one-time, then `streamlit run` works normally)
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Press Y to confirm. Then restart PowerShell.
streamlit run app.py    # now works
```

### Fix C — Use Command Prompt (cmd.exe) instead of PowerShell
```cmd
cd C:\path\to\your\project
pip install -r requirements.txt
streamlit run app.py
```

### Fix D — If `python` is not recognised either
```powershell
# Python not in PATH — use the full path
& "C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
```

---

## 2  Complete Installation (Windows)

```powershell
# 1. Install Python 3.11+ from python.org (tick "Add to PATH")

# 2. Open PowerShell in your project folder
cd C:\Users\YourName\Desktop\TIP-ESG-Platform

# 3. (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # if this fails, use Fix B above first
# OR use cmd.exe: .\venv\Scripts\activate.bat

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Run the Streamlit app
python -m streamlit run app.py
# Opens automatically at http://localhost:8501
```

### requirements.txt (complete)
```
streamlit>=1.35.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
openpyxl>=3.1.0
jupyter>=1.0.0
notebook>=7.0.0
nbformat>=5.9.0
requests>=2.31.0
msal>=1.24.0
python-dotenv>=1.0.0
```

---

## 3  Jupyter Notebooks (formula_engine + consolidation)

```powershell
# Install Jupyter
pip install jupyter notebook

# Open formula_engine.ipynb
python -m jupyter notebook formula_engine.ipynb

# Open consolidation analysis
python -m jupyter notebook consolidation_analysis.ipynb

# Or open all notebooks in browser
python -m jupyter notebook
```

> **Tip:** VS Code users — install the "Jupyter" extension and open .ipynb files directly in VS Code. No command line needed.

---

## 4  OneDrive / SharePoint Setup (E5 Plan)

dss+ is on **Microsoft 365 E5** with **1TB OneDrive per user**.
The platform can use SharePoint as its storage backend — no extra cloud accounts needed.

### What you get with E5 + SharePoint:
| Feature | Detail |
|---|---|
| Storage | 1TB OneDrive per user + SharePoint site quota |
| Data protection | Purview DLP — auto-labels sensitive ESG data |
| Access control | Role-based (analyst / manager / client read-only) |
| Collaboration | Multiple team members see live file updates |
| Audit trail | Every upload/download logged in Microsoft 365 compliance center |
| No extra cost | Included in E5 license already paid |

### For the MOCK (personal account, no Azure AD app needed):
The `storage.py` file has a `MockStorage` class that mirrors all folder structure locally.
It works without any credentials — just run the app and files go to `./mock_storage/`.

### For PRODUCTION (SharePoint):
1. Ask IT to register an Azure AD App in `portal.azure.com`
2. Grant permissions: `Files.ReadWrite.All` and `Sites.ReadWrite.All`
3. Create a `.env` file in the project root:
```
AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_CLIENT_SECRET=your-secret-here
AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
SHAREPOINT_SITE=consultdss.sharepoint.com:/sites/TIP-ESG
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### SharePoint folder structure (auto-created):
```
TIP-ESG-Data (document library)
├── 01_Templates_Raw/
│   ├── Bridgestone/
│   ├── VerdaTyres/
│   └── ...
├── 02_Validated/           ← after dss+ analyst approval
├── 03_Consolidated/        ← master consolidated workbook
├── 04_Reports/             ← generated populated reports
└── 99_Archive/             ← superseded files (never deleted)
```

---

## 5  Full Project File Structure

```
TIP-ESG-Platform/
│
├── app.py                          ← Streamlit frontend (all 5 pages)
├── formula_engine.py               ← Core ESG calculations (used by app.py)
├── storage.py                      ← OneDrive / SharePoint file operations
├── llm_client.py                   ← Azure OpenAI (Copilot) integration
├── requirements.txt
├── .env                            ← (not committed to git — keep secret)
│
├── formula_engine.ipynb            ← Notebook: formulas + analysis + charts
├── consolidation_analysis.ipynb    ← Notebook: full benchmarking analysis
│
├── mock_storage/                   ← Created automatically when MockStorage runs
│   ├── 01_Templates_Raw/
│   ├── 02_Validated/
│   ├── 03_Consolidated/
│   ├── 04_Reports/
│   └── 99_Archive/
│
└── dummy_data/
    ├── TEMPLATE_VerdaTyres_2021.xlsx
    └── (other dummy files)
```

---

## 6  Notes Log (decisions confirmed this session)

| # | Decision | Detail |
|---|---|---|
| 1 | **LLM: Azure OpenAI (Copilot backend)** | dss+ is Copilot partner on E5 plan. Uses same Azure OpenAI endpoint. Zero data retention guaranteed in Enterprise Agreement. |
| 2 | **Storage: SharePoint via Microsoft Graph API** | 1TB OneDrive/user already included in E5. Purview DLP auto-protects sensitive ESG data. No new cloud account needed. |
| 3 | **Multi-user collaboration** | SharePoint document library supports live file sync for multiple dss+ analysts simultaneously. |
| 4 | **Data never touches LLM raw** | `llm_client.py` enforces: only derived KPI summaries sent, company name anonymised, raw Excel never transmitted. |
| 5 | **Frontend: Streamlit (not just HTML mock)** | Same 5-page structure. Use `python -m streamlit run app.py` in PowerShell. |
| 6 | **Notebooks replace formula_engine.py for analysis** | `formula_engine.ipynb` = live calculation + charts. `consolidation_analysis.ipynb` = full benchmarking. `formula_engine.py` still exists for Streamlit imports. |
| 7 | **Dummy data pending manager approval** | VerdaTyres Corp template created. Populated sheet still to be confirmed. |
| 8 | **E5 plan confirmed** | Evans confirmed E5 + 1TB OneDrive. Microsoft Purview Information Protection available. |
| 9 | **Industry: Tire manufacturing (TIP)** | Bridgestone, Yokohama, Michelin, etc. 10 members, ~90% global sales. |
| 10 | **Mock uses local MockStorage** | No credentials needed for demo. Real deployment uses `StorageClient` with Azure AD app. |
