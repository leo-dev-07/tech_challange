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

## Docker

Build and run the generator in a container:

```bash
docker build -t prospectiq-gen .
docker run -v $(pwd)/output:/app/output prospectiq-gen --seed 42 --companies 10 --outdir /app/output
```

Or with custom arguments:

```bash
docker run -v $(pwd)/output:/app/output prospectiq-gen --seed 100 --companies 20 --industries "SaaS,Healthcare"
```

Outputs are written to the local `output/` directory (mounted volume).

## Code overview

This section describes the main classes and generator methods so you can quickly
understand and extend the code.

- **Company**: dataclass representing a target account. Fields include
	`company_id`, `industry`, `employee_count`, `annual_revenue_usd`, and
	`context_note` (a short description of pains/themes).
- **Contact**: person at a company. Includes `contact_id`, `company_id`,
	`full_name`, `title`, `seniority`, `email`, and a `relationship_note`.
- **SalesRep**: internal sales rep with `tier` (Top/Good/Average/Underperformer)
	and a quarterly `quarter_deals_closed_target` used to simulate rep behavior.
- **Deal**: opportunity attached to a company and a primary contact. Contains
	`stage`, `health`, `value_usd`, `opened_at`, `expected_close_at`, `closed_at`,
	and `loss_reason` when applicable.
- **Email**: threaded email messages with `direction` (outbound/inbound),
	`sequence_index`, `sentiment`, `trackers`, and `reply_latency_hours`.
- **Meeting**: scheduled/actual meeting records with attendees, notes, outcome,
	sentiment and follow-up actions.

Generator methods (in `app/core.py`):

- `Generator(seed=None)`: construct the generator; pass `seed` for deterministic
	outputs (seeds `random` and `Faker`).
- `gen_companies(n, industries)`: create `n` `Company` objects across the given
	industries. Employee count and revenue are sampled to later scale deal values.
- `gen_contacts(companies, per_company)`: produce contacts for each company.
- `gen_reps(n)`: produce a small pool of sales reps with tiered targets.
- `gen_deals(companies, contacts, reps)`: create deals; each company can have
	0-2 deals. Deal values are roughly scaled by company size.
- `gen_emails(deals, contacts, reps)`: create email threads per deal. The
	number of messages, sender/recipient direction, and reply latency vary by
	deal stage and health to simulate realistic cadence.
- `gen_meetings(deals, contacts, reps)`: generate meetings following a typical
	sales cadence (Discovery -> Demo -> etc.) scheduled relative to `opened_at`.

The code is intentionally small and easy to extend. If you want stricter
validation, more realistic templates, or multi-threaded generation, I can add
those next.
