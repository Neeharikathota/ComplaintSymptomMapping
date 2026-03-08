# Symptom Code Mapper — Setup & Run Guide

## What it does
Uploads an Excel workbook, reads two sheets (`Investigation Inputs` and
`Symptom Code`), then uses **semantic similarity** (sentence-transformers
`all-MiniLM-L6-v2`) to map each *Investigation Summary* to the best-matching
symptom code. Results are displayed in a colour-coded table and can be
downloaded as Excel.

---

## Prerequisites
- Python 3.10 or later
- pip

---

## 1 · Create a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 2 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2`
> model (~90 MB) on first run. Subsequent runs use the cached copy.

---

## 3 · Run the app

```bash
streamlit run symptom_mapper_app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically.

---

## 4 · Using the app

| Step | Action |
|------|--------|
| 1 | Upload your `.xlsx` file using the file uploader |
| 2 | Preview both sheets in the expandable sections |
| 3 | Adjust the **Confidence threshold** slider in the sidebar if needed |
| 4 | Click **▶️ Run Mapping** |
| 5 | Review the colour-coded results table |
| 6 | Click **⬇️ Download Results as Excel** |

---

## Expected Excel file format

### Sheet: `Investigation Inputs`
| Investigation Summary | … other columns … |
|-----------------------|-------------------|
| Device failed to power on after storage | … |

### Sheet: `Symptom Code`
| Code | Description |
|------|-------------|
| SC-001 | Battery failure |
| SC-002 | Display malfunction |

Column names are auto-detected (case-insensitive keyword match on
`code` / `desc` / `name`). You can override them in the UI.

---

## Confidence threshold guidance

| Score range | Meaning |
|-------------|---------|
| ≥ 0.60 | Strong match |
| 0.35 – 0.59 | Moderate match |
| < 0.35 | Weak / no match → labelled **Others** |

Lower the threshold to be more permissive; raise it to be stricter.
