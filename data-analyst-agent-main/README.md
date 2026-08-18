# Data Analyst Agent

AI agent that answers questions about your data in plain English — no SQL knowledge needed.

---

## What it does

Ask questions in natural language → agent writes and runs SQL → returns the answer.

```text
You:   Which product sells the most?
Agent: Phone — 75 units sold across all cities.

You:   Total revenue by city?
Agent: Danang 2.4B  |  Hanoi 3.0B  |  HCMC 5.7B
```

If AWS credentials are set (see **Configuration**), the agent automatically switches to the **SmartCity pipeline** dataset (vehicle, weather, emergency data on S3) instead of the local sales table — same interface, no code change needed:

```text
You:   What is the average speed of electric vehicles today?
Agent: 78.3 km/h average across today's electric vehicle trips (date=2026-06-17).
```

---

## Stack

| Layer              | Tool                                 |
| ------------------ | ------------------------------------ |
| LLM                | Groq — Llama 3.3 70B                 |
| Agent framework    | LangChain                            |
| Database           | DuckDB (local file or S3 via httpfs) |
| CLI                | Rich                                 |
| Retry / resilience | Tenacity                             |

---

## Project structure

```text
data-analyst-agent/
├── src/
│   ├── agent/
│   │   └── agent.py          # LangChain agentic loop + Groq integration
│   ├── tools/
│   │   ├── sql_tool.py       # query_sql and list_tables tools (local DuckDB + S3/httpfs)
│   │   └── file_tool.py      # extensible file ingestion (WIP)
│   └── logging_config.py     # JSON structured logger
├── tests/
│   ├── conftest.py            # stubs for all external deps (no live services needed)
│   ├── test_agent.py          # 43 cases — agent loop, retry, fallback XML
│   ├── test_sql_tool.py       # 40 cases — sanitize, S3/local conn lifecycle, overflow rewrite
│   └── test_logging_config.py # 26 cases — JsonFormatter, extra fields, LOG_LEVEL
├── data/sample/
│   └── warehouse.db           # auto-created on first run (local mode only)
├── main.py                    # entry point
├── pyproject.toml
└── .env.example
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/minnobug/data-analyst-agent.git
cd data-analyst-agent
pip install -e .
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your Groq API key — free at console.groq.com
```

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO
```

Leave `AWS_ACCESS_KEY` / `AWS_SECRET_KEY` / `AWS_BUCKET_NAME` unset to run in local mode (sample sales data, auto-seeded). Fill them in to point the agent at the SmartCity S3 pipeline instead — see **Configuration**.

### 3. Run

```bash
python main.py
```

---

## Run tests

```bash
# All 109 tests — no live services or API keys needed
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

| Env var           | Default                    | Description                                                         |
| ----------------- | -------------------------- | ------------------------------------------------------------------- |
| `GROQ_API_KEY`    | required                   | Groq API key                                                        |
| `GROQ_MODEL`      | `llama-3.3-70b-versatile`  | Model to use                                                        |
| `WAREHOUSE_DB`    | `data/sample/warehouse.db` | Path to local DuckDB file (used when AWS vars below are not set)    |
| `LOG_LEVEL`       | `INFO`                     | `DEBUG` / `INFO` / `WARNING`                                        |
| `AWS_ACCESS_KEY`  | —                          | Enables SmartCity/S3 mode when set together with the two vars below |
| `AWS_SECRET_KEY`  | —                          | AWS secret key for S3 access                                        |
| `AWS_BUCKET_NAME` | —                          | S3 bucket holding the SmartCity pipeline's `refined/` Parquet data  |
| `AWS_REGION`      | `ap-southeast-1`           | AWS region for the bucket                                           |

When all three AWS vars are present, the agent connects via DuckDB's `httpfs` extension to the SmartCity tables (`vehicle_data`, `gps_data`, `traffic_data`, `weather_data`, `emergency_data`) and falls back to the local warehouse automatically if the S3 data is missing or unreachable.

---

## Roadmap

* Natural language → SQL
* DuckDB local warehouse
* Connect to SmartCity pipeline (S3 Parquet via DuckDB httpfs)
* Groq rate-limit retry with tenacity
* JSON structured logging
* 109 unit tests, CI on GitHub Actions
* File ingestion tool (CSV, Parquet upload)
* Streamlit web UI

---
