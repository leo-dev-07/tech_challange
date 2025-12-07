# ProspectIQ Demo Data Generator

This repository provides a deterministic demo data generator for ProspectIQ (companies → contacts → deals → emails → meetings). It produces JSON outputs suitable for demos and validation.

## Quick Start: Data Generation & RAG Chat

### 1. Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

### 2. Generate Demo Data

```powershell
python -m app.generate --seed 42 --companies 500 --outdir output
```

Output files: `output/companies.json`, `output/contacts.json`, `output/deals.json`, `output/emails.json`, `output/meetings.json`, `output/sales_reps.json`

### 3. Run RAG Chat Application

```powershell
python -m app.chat
```

Then query anything about your data:
```
Query: How many SaaS companies?
Query: Show all demo meetings
Query: Find IC contacts
Query: What discovery calls are scheduled?
```

## RAG System - Fully Dynamic Indexing

The included RAG (Retrieval-Augmented Generation) system features **completely dynamic indexing** - no hardcoding needed!

### How It Works

1. **Automatic Field Discovery**: Indexes ALL fields from ALL data entities
2. **Any Query**: Query by any field value (company name, meeting title, contact seniority, etc.)
3. **Zero Code Changes**: Add new fields or entities without modifying code
4. **LLM Context**: Provides comprehensive context to Groq/OpenAI for natural responses

### Architecture

```
Your Query
    ↓
Dynamic QueryAnalyzer (detects entities and filters)
    ↓
RAGStore (indexes all fields dynamically)
    ↓
Semantic Search (TF-IDF or Sentence-Transformers)
    ↓
Dynamic StatisticsBuilder (builds context from retrieved data)
    ↓
LLM (Groq llama-3.3-70b-versatile or OpenAI)
    ↓
Natural Language Response
```

### What's Indexed

**3,961 Documents Total:**
- 500 companies (all fields: id, name, industry, revenue, etc.)
- 1,500 contacts (all fields: id, name, title, seniority, etc.)
- 489 deals (all fields: id, stage, value, etc.)
- 500 emails (sample, all fields)
- 966 meetings (all fields: id, title, dates, outcome, etc.)
- 6 sales reps (all fields)

### Example Queries

```
"How many meetings are there?" → Returns total + distributions
"Show all discovery calls" → Returns meetings with "Discovery Call" title
"SaaS companies" → Returns companies with industry=SaaS
"Find IC contacts" → Returns contacts with seniority=IC
"What's our deal pipeline?" → Returns deal statistics by stage
```

### Adding New Data

The system automatically works with new fields or entities - **no code changes required!**

1. Add data to `output/*.json` files
2. Update `DataStore` in `app/data_loader.py` (1 line)
3. Done! Everything else adapts automatically

### Testing

```powershell
# Test dynamic indexing
python test_dynamic_index.py

# Test end-to-end queries
python test_end_to_end.py
```

## Original Documentation

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
- `Seed`: A seed is an initial value used to initialize a random number generator.
    Once you set a seed, all subsequent "random" values  become deterministic — the same seed produces the same sequence of random numbers every time.
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

## Query Agent with RAG (Retrieval-Augmented Generation)

A new agent app allows you to chat with the generated data. The agent uses a **RAG (Retrieval-Augmented Generation)** architecture to understand queries without hardcoded rules:
- **Semantic search** finds relevant companies, contacts, deals, and emails based on meaning (not just keywords)
- **LLM-augmented responses** (optional): if you provide an OpenAI or Groq API key, responses are generated by an LLM with the retrieved data as context
- **Local fallback mode**: works without external APIs using TF-IDF semantic search

### Setup

1. Copy `.env.example` to `.env` (optional, for LLM mode) to use with your own credentials:

```powershell
Copy-Item .env.example .env
# Edit .env and add your API keys (or leave blank for local-only mode):
# OPENAI_API_KEY=sk-...  (for OpenAI Chat Completions)
# GROQ_API_KEY=gsk-...   (for Groq via langchain)
```

2. Dependencies are in `requirements.txt` (TF-IDF and optional sentence-transformers):

```powershell
pip install -r requirements.txt
```

### Run the agent

```powershell
python -m app.chat
```

This launches an interactive chat session. Ask natural questions like:
- "What companies are in the Healthcare industry?"
- "Tell me about Daniel Massey"
- "Show me deals at the Demo stage"
- "Who are the top contacts at Acme Corp?"
- "What's the average employee count in Technology?"

**No API key needed** — the agent works in local mode using TF-IDF semantic search.

### Architecture

**RAG Pipeline:**

- **`app/rag.py`**: `RAGStore` class handles semantic search:
  - Converts dataset records (companies, contacts, deals, emails) into text documents
  - Uses **TF-IDF** (default, always available) or **sentence-transformers** (optional, better quality)
  - Builds a searchable index using scikit-learn or FAISS
  - Returns top-K most relevant records for any user query

- **`app/data_loader.py`**: `DataStore` loads generated JSON files and provides indexing

- **`app/agent.py`**: Agent orchestration:
  1. User query arrives → `_gather_context_snippets` calls RAG to retrieve relevant records
  2. If LLM is available (Groq or OpenAI): send query + snippets to LLM for a natural answer
  3. If LLM is unavailable: format snippets into a readable response (companies, contacts, deals, etc.)

- **`app/chat.py`**: Interactive CLI

**Two Modes:**

| Mode | Setup | Quality | Speed |
|------|-------|---------|-------|
| **Local (TF-IDF)** | No API key | Good | Fast |
| **LLM-augmented (OpenAI/Groq)** | Add `OPENAI_API_KEY` or `GROQ_API_KEY` to `.env` | Excellent | Depends on API |

### Example session

```
You: What companies are in the Technology sector?
Agent: **Companies found:**
  - TechCorp Inc (Technology, 250 employees)
  - DataFlow Systems (Technology, 120 employees)

**Contacts found:**
  - Alice Johnson: VP Sales (C-Level)
  - Bob Smith: Solutions Architect (Manager)

You: Tell me about Daniel Massey
Agent:
**Contacts found:**
  - Daniel Massey: Manager (IC)

You: Exit
```

### Extending the RAG

To handle custom data or improve search quality:

1. **Add more data sources** to `rag.py` in the `_build_index` method (e.g., meetings, support tickets)
2. **Switch to better embeddings** if network is available:
   - Uncomment/enable `sentence-transformers` in `requirements.txt`
   - RAG will automatically fall back to TF-IDF if the model download fails
3. **Tune search parameters**:
   - Modify `top_k` (number of results) in `_gather_context_snippets`
   - Adjust TF-IDF `max_features` and `stop_words` in `rag.py`

---
