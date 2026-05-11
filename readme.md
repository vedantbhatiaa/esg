# TIP ESG Platform

ESG KPI reporting system for the WBCSD Tire Industry Project, built with Streamlit.

---

## Requirements

- Python 3.10 or higher — download from [python.org](https://python.org) (**tick "Add Python to PATH"** during install)

---

## Setup & Run (Windows — Command Prompt)

Open Command Prompt (`Win + R` → type `cmd` → Enter), then run these commands one by one:

```cmd
cd C:\path\to\project-folder

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Place `CONSOLIDATED_DUMMY_2009_2023.xlsx` in the project root folder, then build the master database:

```cmd
python build_esg_master.py
```

Run the app:

```cmd
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Login Credentials

| Role | Email | Password |
|---|---|---|
| dss+ Analyst (all access) | `employee@consultdss.com` | `demo1234` |
| VerdaTyres Corp | `verdatyres@tip-reporting.com` | `demo1234` |
| AlphaTread Ltd | `alphatread@tip-reporting.com` | `demo1234` |
| BetaRubber Inc | `betarubber@tip-reporting.com` | `demo1234` |
| GammaTire SA | `gammatire@tip-reporting.com` | `demo1234` |
| DeltaGrip GmbH | `deltagrip@tip-reporting.com` | `demo1234` |

---

## Project Structure

```
project/
├── app.py                                    <- main Streamlit app
├── formula_engine.py                         <- ESG calculations
├── data_loader.py                            <- loads consolidated data
├── build_esg_master.py                       <- run once to build master CSV
├── llm_client.py                             <- Azure OpenAI integration
├── local_storage.py
├── storage.py
├── requirements.txt
├── CONSOLIDATED_DUMMY_2009_2023.xlsx         <- place here before running
└── data_storage/
    └── raw/                                  <- master CSVs saved here
```

---

## After Adding New Data

Whenever you update the consolidated Excel, re-run:

```cmd
python build_esg_master.py
```

---

## Common Issues

| Error | Fix |
|---|---|
| `python` not recognised | Reinstall Python and tick "Add to PATH" |
| `FileNotFoundError` on Excel | Place `CONSOLIDATED_DUMMY_2009_2023.xlsx` in the project root |
| App shows N/A everywhere | Run `python build_esg_master.py` first |
| `Permission denied` when saving | Close the CSV file in Excel, then try again |

---

## Optional — Azure OpenAI (AI Readiness page)

Create a `.env` file in the project root:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Without this the AI page still works in mock mode — no key required for local testing.