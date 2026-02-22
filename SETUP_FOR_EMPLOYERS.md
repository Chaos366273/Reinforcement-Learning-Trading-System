# Setup for Employers / Reviewers

This folder contains the **PPO Trading Agent** quant research project. **Yes — you can run this code.** Follow the steps below; the main entry point to train and evaluate the agent is **`PPO_fixed.py`**.

**A trained model is included** (`ppo/saved_models/ppo_trading_agent.zip`) for the report **UNIFIED_REPORT_2026-02-21_18-55-56**. You can run evaluation with it, or train from scratch using the steps below.

---

## Can they run this code?

**Yes.** Anyone with Python 3.10+, the dependencies in `requirements.txt`, and (for full training) market data access (Alpaca or Yahoo) can run the project. Run the steps below **in order**. The correct way to run the agent is to execute **`PPO_fixed.py`** after preparing data and the BC model.

---

## How to run the code (correct order)

You **must** run scripts in this order so the agent has data and a BC-initialized policy:

| Step | Command | Purpose |
|------|---------|--------|
| 1 | `python generate_csv_data.py` | Build daily CSV dataset in `ppo/saved_data/csv_data/` |
| 2 | `python precompute_calculations.py` | Pre-compute VWAP, returns, volatility, etc. (speeds up training) |
| 3 | `python run_bc.py` | Train Behavioral Cloning model; saves `ppo/saved_models/ppo_bc_init.pt` |
| 4 | **`python PPO_fixed.py`** | **Main entry point:** train PPO agent, run evaluation, generate reports |

**To actually run the agent (train + evaluate), you run `PPO_fixed.py`.** It orchestrates data loading, environment creation, PPO training, validation, and unified/SHAP report generation. Do not skip steps 1–3 or the run will fail (missing data or BC init).

---

## Quick start (copy-paste)

From the **repo root** (the folder containing `PPO_fixed.py`):

```powershell
# 1. Virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Dependencies
pip install -r requirements.txt

# 3. Data (one-time; needs Alpaca or Yahoo)
python generate_csv_data.py
python precompute_calculations.py

# 4. BC then PPO — this is how you run the agent
python run_bc.py
python PPO_fixed.py
```

On Linux/macOS use `source .venv/bin/activate` instead of the PowerShell activate line.

---

## What each script does

- **`PPO_fixed.py`** — Main script. Runs the full pipeline: load data → create env → train PPO (with optional BC init) → evaluate → write `ppo/daily_reports/UNIFIED_REPORT_*.md` and validation artifacts. **This is the script that runs the agent.**
- **`run_bc.py`** — Builds expert dataset and trains BC; run once (or after changing training window) before `PPO_fixed.py` if using BC initialization.
- **`generate_csv_data.py`** — Downloads/builds daily CSV into `ppo/saved_data/csv_data/`.
- **`generate_intraday_csv.py`** — Optional; adds 1m/5m/30m/1h CSVs for multi-timeframe.
- **`precompute_calculations.py`** — Fills `ppo/saved_data/calculation_cache/`; run after (re)generating CSV.

---

## What's included

- **Source:** `PPO_fixed.py`, `run_bc.py`, `config.py`, data scripts, and the `ppo/` package (models, utils, evaluation, training, data, tests, docs).
- **Sample outputs:** `ppo/daily_reports/` has two unified reports + an improvement summary; `ppo/explanations/` has a sample SHAP report.
- **Docs:** `README.md` (full project docs), `ppo/docs/` (reward changelog, performance notes).

---

## Notes

- **API keys:** `config.py` may contain Alpaca credentials for paper trading. For your own runs, use environment variables or your own keys.
- **Data:** Training expects CSVs in `ppo/saved_data/csv_data/`. Create that folder if needed; it will be populated by `generate_csv_data.py`.
- **Tests:** From repo root: `python -m ppo.tests.test_quant_improvements_2026_02_09` and `python -m ppo.tests.test_bull_bear_and_intraday`.

For full architecture, phases, and troubleshooting, see **README.md**.
