# ProspectIQ Demo Data Generator

This repository provides a deterministic demo data generator for ProspectIQ (companies → contacts → deals → emails → meetings). It produces JSON outputs suitable for demos and validation.

Quick start

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the generator (example):

```powershell
python -m app.generate --seed 42 --companies 10 --industries SaaS,Healthcare --outdir output
```

3. Validate outputs (basic checks):

```powershell
python -m app.validate --indir output
```

Files generated (JSON): `companies.json`, `contacts.json`, `sales_reps.json`, `deals.json`, `emails.json`, `meetings.json`.

See `app/generate.py` for CLI options and `app/core.py` for model/logic.
