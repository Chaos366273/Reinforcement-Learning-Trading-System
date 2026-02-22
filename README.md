# 🚀 PPO Trading Agent - Super Agent Development

## 📊 **Project Overview**

This project implements a **Proximal Policy Optimization (PPO) trading agent** that learns to trade stocks using reinforcement learning. The agent has been optimized through multiple phases to achieve **super agent status** with realistic Alpaca trading costs and intelligent reward structures.

### **Training setup (included model & data)**

- **Timesteps:** The agent is trained on **50,000 timesteps** (baseline phase; scaling to higher timesteps is configurable in `ppo/utils/dynamic_config.py`).
- **Indicators:** The model uses **15 features/indicators** (e.g. close, returns, RSI, MACD, MACD signal, Bollinger z-score/width, ATR, volume ratio, time-of-day, SPY context). In intraday mode, 14 features per bar are used (see **Feature set & noise** below).
- **Universe (companies/symbols):** Training and evaluation use the tickers defined in `config.py` → `DATA_CONFIG['tickers']`. The default universe includes **AAPL, MSFT, NVDA, GOOGL, META, AMD, JPM, V, BAC, MA, AMZN, WMT, HD, NKE, KO, COST, JNJ, UNH, PFE, LLY, ABBV, XOM, CVX, VZ, T, DIS, INTC, QCOM, ORCL, CRM** (diverse sectors: tech, financials, consumer, healthcare, energy). You can change the list in `config.py` and regenerate CSV data.

## 🎯 **Current Status: PHASE 7 IN PROGRESS - REWARD TUNING & SCALING (2026-02-20)**

**Note:** README reward/performance snapshots may lag; **canonical config** is `ppo/utils/dynamic_config.py`. Latest run metrics and analysis: see **LOW_RETURN_ANALYSIS_AND_PLAN_2026-02-05.md**, **ANALYSIS_AND_IMPROVEMENT_PLAN_2026-02-05.md**, **REWARD_STRUCTURE_ANALYSIS_2026-02-05.md**, **QUANT_RESEARCH_AGENT_PERFORMANCE_2026-02-09.md**, **REPORT_ERRORS_2026-02-09.md**, **ppo/docs/PERFORMANCE_REFINEMENTS_2026-02-09.md**, **ppo/docs/REWARD_CHANGELOG.md**, and `ppo/daily_reports/UNIFIED_REPORT_*.md`. Plan: **.cursor/plans/super_agent_scaling_plan_47880b09.plan.md**.

#### **2026-02-20 Super Agent Scaling Plan & System Architecture**
- **Super agent scaling plan (2026-02-19)** implemented: market-first **RegimeController** (`ppo/utils/regime_controller.py`) drives BUY/SELL/HOLD reward multipliers and gating thresholds from regime (volatility, bull/bear, trend). Behavior-adaptive logic is tertiary. Report consistency: headline metrics = **deterministic** (paper-trading truth); stochastic stored separately as diagnostic. Collapse diagnostics in unified report: raw BUY%, entropy, gating blocks.
- **Deterministic collapse (agent not trading enough)**: Latest runs show deterministic eval with **HOLD 93–96%**, **BUY 1–2%**, **4–6 trades** while stochastic eval shows **30+ trades** and positive return. **Root cause**: Policy learned to prefer HOLD because **RegimeController reduces HOLD penalty in volatile regimes** (`hold_penalty_mult *= 0.75` when `avg_volatility >= 2%`). During training, many steps are “volatile” → HOLD is less penalized → policy converges to argmax HOLD. Gating is **not** the blocker (report: Gating blocked = 0). See **Troubleshooting → Agent not trading enough** below.
- **Recommended fixes (no code change without your approval)**: (1) In `ppo/utils/regime_controller.py`, cap or remove the volatile-regime HOLD penalty reduction (e.g. do not go below 1.0x, or use 0.90x min instead of 0.75x). (2) Strengthen eval HOLD penalty when deterministic BUY% is very low (collapse guardrail in eval or config). (3) Optional: increase base `hold_penalty` / `hold_penalty_no_positions` in `dynamic_config.py` so HOLD is less attractive during training.

**Known issues (2026-02-05):** (1) ~~PerformanceEvaluator import broke adaptive LR callback~~ **FIXED** – added `PerformanceEvaluator` wrapper in `ppo/evaluation/performance_evaluator.py`. (2) NaN prices in data for some tickers (env fallback to 0.01) – trace to CSV/precompute and fix at source. (3) SHAP phase calls `env.reset()` and runs extra rollout → "UNEXPECTED RESET" and SELL-with-no-positions log noise.

#### **2026-02-09 Quant Improvements, Adaptive LR Fix & Performance**
- **Adaptive LR (poor) was reducing LR**: Report showed LR 3e-4 → 1e-4 when Sharpe < 0.5 (cap was 1e-4). **Fixed**: `adaptive_lr_poor_max` raised to 5e-4 in `ppo/utils/dynamic_config.py`; in `PPO_fixed.py`, poor branch now uses `new_lr = max(current_lr, min(poor_lr_max, current_lr * factor))` so LR never drops when intending to boost. See **REPORT_ERRORS_2026-02-09.md**.
- **SELL signal_alignment_bonus tied to outcome**: In `ppo/models/trading_env.py`, SELL gets full signal_alignment_bonus only when **realized_pnl > 0**; when pnl ≤ 0 bonus = 0; when 0 < pnl ≤ 0.001 bonus at 25%. Rewards profitable, signal-aligned SELLs more.
- **Activity bonuses further reduced (PnL over activity)**: `trading_frequency_bonus` 0.06→0.03, `progressive_bonus_per_trade` 0.005→0.002, `progressive_bonus_max` 0.15→0.08, `exploration_bonus` 0.15→0.10 in `ppo/utils/dynamic_config.py`.
- **Hold-winner bonus strengthened**: `hold_winner_bonus` 0.03→0.05 in `ppo/utils/dynamic_config.py`.
- **Per-step config cache (performance)**: In `ppo/models/trading_env.py`, config is fetched once at the start of each `step()` and cached in `self._dcfg`; new `_cfg()` returns it. Replaces 100+ `get_dynamic_config()` calls per step with one. Logic unchanged. See **ppo/docs/PERFORMANCE_REFINEMENTS_2026-02-09.md**.
- **Step-logging default**: Default `log_every_n_steps` in env increased from 200 to 400 (fewer I/O; overridable via config).
- **Tests**: `ppo/tests/test_quant_improvements_2026_02_09.py` verifies adaptive LR math, config defaults (poor_max ≥ 5e-4, activity bonuses, hold_winner), SELL bonus formula, and callback source. Run: `python -m ppo.tests.test_quant_improvements_2026_02_09`.
- **Docs**: **QUANT_RESEARCH_AGENT_PERFORMANCE_2026-02-09.md** (analysis and solutions), **REPORT_ERRORS_2026-02-09.md** (report issues), **ppo/docs/PERFORMANCE_REFINEMENTS_2026-02-09.md** (speed refinements).

#### **2026-02-08 Evaluator & Report Fixes**
- **Alpha/Beta = 0 in reports**: Root cause was all-zero benchmark series (`benchmark_returns.csv`). **Fixed** in `ppo/evaluation/performance_evaluator.py`: benchmark return is now built from env (SPY step returns when SPY in universe, else equal-weighted universe return); alpha/beta computed only when benchmark has non-trivial variance; report shows **N/A** when benchmark invalid (`ppo/evaluation/unified_report_generator.py`).
- **Max consecutive wins/losses = 0**: These were never computed. **Fixed**: evaluator now computes `max_consecutive_wins` and `max_consecutive_losses` from closing-trade PnL (SELL/AUTO_SELL/REBALANCE_SELL/STOP_LOSS) and writes them to results; unified report displays them in Trading Behavior Analysis.
- **Sharpe tracking**: Stored in `ppo/daily_reports/unified_report_*.json` and `ppo/saved_data/validation/backtest_results.json` under `sharpe_ratio`. Compare runs by timestamp; annualization currently uses sqrt(252) (daily assumption—intraday bar eval may want different scaling for interpretation).

#### **2026-02-05 Reward & Dynamic Scaling Update**
- **📋 Reward structure doc**: `REWARD_STRUCTURE_ANALYSIS_2026-02-05.md` – full breakdown of BUY/SELL/HOLD and post-step rewards; why returns stayed flat (activity vs outcome).
- **⚖️ Activity bonuses reduced (outcome over activity)**: `trading_frequency_bonus` 0.25→0.12, `progressive_bonus_per_trade` 0.02→0.01, `progressive_bonus_max` 0.50→0.25, `exploration_bonus` 0.50→0.25. Eval hold: `eval_hold_penalty` -0.04→-0.03, `eval_hold_penalty_growth_threshold` 0.01→0.005.
- **📈 Dynamic reward scaling (SELL)**: Bonuses and penalties now scale by market/portfolio context via `_get_dynamic_reward_context()` (regime, vol_ratio, portfolio_return, current_drawdown). Applied to: **trading frequency & progressive** (0.7x in mean_reversion), **exploration** (1.2x in drawdown >5%, 0.9x when portfolio return >1%), **signal alignment** (1.2x trending, 0.85x mean_reversion), **gain-threshold bonuses** (vol_scale 0.7–1.2 by vol_ratio), **sell hold-time bonus** (regime mult), **quick-flip penalty** (1.2x trending, 0.8x mean_reversion), **loss penalty** (up to 1.5x when drawdown >2%). Context cached 10 steps; cleared on reset.

### **✅ LATEST ACHIEVEMENTS (Phase 7 - LATEST UPDATE):**
- **🧠 TCN Memory Architecture**: Upgraded from MlpPolicy to TCN (Temporal Convolutional Network) policy for temporal memory - more powerful than LSTM, faster on CPU
- **⚡ BC Regularization Optimization**: 99.9% reduction in computation - checks every 1000 steps (was every step), 90% lighter penalty (5% LR reduction vs 50%), weight reduced from 0.3 to 0.03 (10x lighter)
- **📊 Evaluation Frequency Optimization**: Increased from 50k to 100k steps (50% reduction in evaluation overhead)
- **📝 Progress Logging Optimization**: Increased log frequency from 10k to 25k steps (60% reduction in I/O overhead)
- **🔄 Adaptive Learning Rate**: Bidirectional - maintains higher LR (1.5x) when Sharpe > 2.0 AND Return > 3%; boosts LR when poor (Sharpe/return below thresholds). **2026-02-09**: Poor branch now always *increases* LR (never reduces): `adaptive_lr_poor_max` 5e-4, formula `new_lr = max(current_lr, min(poor_lr_max, current_lr * factor))`
- **🔄 Adaptive Entropy Coefficient**: Increases entropy (cap 0.04) when trade count < 50
- **🔄 Self-Correcting System**: Automatically resets when performance/trade conditions normalize
- **🔄 Dynamic LR Schedule**: Real-time config reading, adjustments take effect immediately
- **🔄 Problem Solved**: Prevents over-conservatism at 300K+ timesteps (was causing 78% return drop)
- **💰 Dynamic Market-Condition Rewards**: Base rewards scale 0.7x-1.6x based on volatility, market trend, and regime
- **🎯 Reward Config Sync**: base_trading_incentive, sell_base_incentive, exploration_bonus sync from dynamic_config
- **⚖️ Reward Balance**: Base rewards (BUY 0.40, SELL 0.40, exploration 0.10); eval hold penalty -0.03; activity bonuses further reduced (2026-02-09: trading_frequency 0.03, progressive 0.002/0.08); SELL rewards scale by regime/vol/drawdown; **SELL signal_alignment_bonus** full only when realized PnL > 0 (2026-02-09)
- **🚀 Signal Gating**: High confidence >0.25, medium >0.10 (hardcoded in env); low-confidence cooldown-based (not fixed % random)
- **📈 SELL Rewards**: sell_base_incentive 0.48, sell_profit_taking_bonus 0.50, sell_bull_market_boost 0.08
- **⚖️ Balanced Loss Penalties**: buy_loss_penalty_cap and sell_loss_penalty_cap both -1.0
- **🐂 Gentle Market Multipliers**: Bull/Bear market adjustments gentle (5-10% tilt)
- **💪 Stop Loss**: Base 10%, trailing 8% (then adjusted by volatility + ATR + regime + signals)
- **📊 RL Rewards**: Technical-only for RL (technical_signal_weight=1.0, fundamental_signal_weight=0.0; fundamental reserved for supervised networks later)
- **🏃 Per-step config cache (2026-02-09)**: Env caches `get_dynamic_config()` once per step (`_cfg()`); reduces 100+ calls/step to one for faster execution without logic change
- **🧪 Quant improvement tests (2026-02-09)**: `ppo/tests/test_quant_improvements_2026_02_09.py`; `hold_winner_bonus` 0.05

### **✅ PREVIOUS ACHIEVEMENTS (Phase 6):**
- **⚡ 10-20x Performance Boost**: Training now 10-20x faster (5 seconds/step → 0.25-0.5 seconds/step)
- **🎯 Advanced Dynamic Stop Loss**: Volatility + ATR + Market Regime + Signal Awareness
- **🔄 Signal-Confidence Gating**: Quality over quantity - only high-confidence signals trigger trades
- **⚖️ BUY/SELL Reward Rebalancing**: Equal base incentives, consistent signal-based bonuses
- **📊 Signal-Aware Stop Loss Rewards**: Penalties adjust based on whether signals warned of drop
- **🚀 Fast Vectorized Portfolio Calculations**: NumPy-based with step-level caching
- **💾 Optimized Signal Caching**: Signals cached per step (not per action), 7x fewer calculations
- **📝 Reduced Logging Overhead**: Debug logs every 20th trade (was 5th), 4x faster I/O

### **Previous Achievements (Phase 5):**
- **Reward Structure**: RL rewards technical-only (technical_signal_weight=1.0, fundamental_signal_weight=0.0; fundamental reserved for supervised nets later)
- **BC Regularization**: Preserves BC knowledge for 50k timesteps (prevents unlearning)
- **Signal Amplification**: Strong signals get 1.5x-2.0x multiplier
- **Signal Confirmation**: High-confidence aligned signals get signal_alignment_bonus (config)
- **Hold Penalty Enhancement**: Stronger penalties when signals exist but agent holds
- **Trade Frequency Increase**: 102 trades vs previous 8 (12.75x improvement)
- **Technical Analysis**: Used for RL reward computation (100% weight for signals in RL)
- **Fundamental Analysis**: Reserved for supervised networks later (0% in RL rewards)

#### 🔄 **September 2025 Reward Alignment Update (current branch)**
- Re-aligned BUY/SELL rewards so realized PnL is the primary driver (all incentives gated behind profitable trades; losses receive explicit penalties).
- Simplified strong-signal HOLD penalties: threshold-based, grace period aware, exempt after masked SELL.
- **CRITICAL**: Retrain from scratch with current config (base_trading_incentive 0.50, sell_base_incentive 0.48, loss penalties -1.0) before live validation or TG5 hand-off.

---

## 🔧 **Technical Architecture**

### **Core Components:**
- **Environment**: `FixedTradingEnv` - Custom trading environment with technical/fundamental analysis
- **Agent**: PPO with TCN (Temporal Convolutional Network) Policy - Upgraded from standard PPO for temporal sequence learning
- **Memory Architecture**: TCN with dilations [1, 2, 4, 8] - Agent can remember 15+ past steps (better than LSTM's 10 steps), faster on CPU
- **Data**: 11 tickers with OHLCV features (config.py DATA_CONFIG); fundamental metrics used for optional analysis (RL rewards are technical-only)
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Volume, ATR-based regime detection (used for reward/signals)
- **Fundamental Analysis**: PE, ROE, Debt-to-Equity, Growth (reserved for supervised networks; fundamental_signal_weight=0 for RL)
- **Rewards**: RL rewards use technical signals only (technical_signal_weight=1.0, fundamental_signal_weight=0.0); PnL and other terms per dynamic_config
- **Stop Loss**: Advanced dynamic system - base 10%, trailing 8% (Volatility + ATR + Market Regime + Signal Awareness)
- **Signal Gating**: Three-tier (high >0.25, medium >0.10, low cooldown-based); thresholds hardcoded in _should_gate_trade_by_signal_confidence
- **Observation / features**: Observation dimensionality and feature count from dynamic_config (num_features, observation_space_size); config.py documents 5 basic OHLCV; technical/fundamental used for reward and signal computation. See **Feature set & noise** below.
- **Performance**: 10-20x optimized (signal caching, vectorized calculations, reduced logging, lighter BC regularization)
- **Costs**: Realistic Alpaca transaction fees (0.0024% per trade)

### **Key Files:**
- `PPO_fixed.py` - Main orchestrator script
- `ppo/models/trading_env.py` - Trading environment (per-step config cache via `_cfg()`; regime + gating)
- `ppo/utils/dynamic_config.py` - Centralized configuration
- `ppo/utils/regime_controller.py` - Market-first RegimeController (BUY/SELL/HOLD multipliers, gating thresholds; 2026-02-19 plan)
- `ppo/utils/agent_explainability.py` - SHAP explainability module
- `ppo/evaluation/performance_evaluator.py` - Deterministic + stochastic eval; writes validation artifacts
- `run_bc.py` - Behavioral Cloning model
- `config.py` - Configuration settings
- `generate_csv_data.py` - Generate immutable CSV dataset (one-time, daily)
- `generate_intraday_csv.py` - Generate intraday CSVs (1m, 5m, 30m, 1h) to `ppo/saved_data/csv_data/` (59-day lookback for 5m/30m/1h; Yahoo 60-day limit)
- `precompute_calculations.py` - Pre-compute runtime calculations cache (one-time)
- `ppo/tests/test_quant_improvements_2026_02_09.py` - Tests for adaptive LR, reward/config defaults, SELL bonus formula (2026-02-09)

#### **TCN (Temporal Convolutional Network) Policy**
- **What it is**: TCN replaces the default MlpPolicy for PPO to add temporal memory — the agent sees sequences of past steps (15+ effective steps via dilated convolutions) instead of a single observation.
- **Where it lives**: `ppo/models/tcn_policy.py` (TCNTradingPolicy), `ppo/training/ppo_trainer.py` (create_ppo_agent uses TCN when enabled).
- **Config** (`ppo/utils/dynamic_config.py`): `use_temporal_memory = True` (enable TCN), `temporal_memory_type = "tcn"`, `tcn_hidden_dims = [256, 256, 128, 128]`, `tcn_dilation_rates = [1, 2, 4, 8]`, `tcn_kernel_size = 3`, `tcn_dropout = 0.1`, `tcn_use_residual = True`.
- **Enable/disable**: Set `use_temporal_memory = False` in dynamic_config to fall back to standard MlpPolicy (no temporal memory).
- **Unified reports**: Report generator checks `config.use_temporal_memory` and policy class name to show "TCN used" or "TCN DISABLED".

#### **Feature set & noise (2026-02; reduced)**
- **Intraday bar mode** (when `use_intraday_step=True`): **14 features** per bar in `ppo/data/csv_loader_adapter.py` — close, ret_1, rsi_14, macd, macd_signal, macd_hist, bb_z, bb_width, atr_14, vol_ratio_20, tod_sin, tod_cos, spy_ret_1, spy_rsi_14. **Dropped**: `log_ret_1` (redundant with ret_1), `vol_z_20` (noisy on thin bars). SPY features are zeros when SPY not in universe (dead dimensions until SPY added).
- **Daily path** (when not using intraday bar matrices): canonical list is `BASE_DAILY_FEATURES` (11) plus optional intraday merge columns (4 per interval); missing columns filled with 0 → many zeros if intraday CSVs absent or partial, which can add noise.
- **Indicator availability (Price Trends & Liquidity)**  
  - **Stock momentum**: ✅ Available (e.g. `ret_1`, `price_momentum`, `short_momentum_5bar` in feature_engineer; momentum in rewards in `trading_env`).  
  - **Industry momentum**: ❌ Not available (no industry/sector grouping or sector index returns).  
  - **Short-term reversal**: ⚠️ Partial (candlestick reversal patterns in `technical_indicators`; no explicit 1‑month reversal factor).  
  - **Market value**: ❌ Not available (no market cap / size factor).  
  - **Dollar volume**: ⚠️ Derivable (volume × price in data; not a named feature; can add as `dollar_volume` or use volume + close).  
  - **Bid-ask spread**: ⚠️ Proxy only (`spread_proxy` = (high−low)/close and `spread_z` in `feature_engineer._create_lob_proxy_features`; no real L2 bid/ask unless you add a feed).
- **If features feel too noisy**: (1) Redundancy already reduced (log_ret_1, vol_z_20 dropped). (2) When SPY is not in universe, spy_ret_1/spy_rsi_14 are zeros. (3) Use SHAP reports (`ppo/explanations/shap_report_*.md`) to drop low-importance features. (4) Try `include_intraday=False` in daily path to use only 11 base daily features.

---

## 📊 **Data Pipeline & Performance Optimization**

### **✅ CSV-Based Data Architecture (NEW - 2026-01-25)**

**Immutable CSV Pipeline** - Industry-standard architecture for deterministic, corruption-free training:

#### **Features:**
- ✅ **Immutable CSV**: Data frozen once, never changes (hash-verified)
- ✅ **Per-Ticker Indicators**: Calculated separately (prevents date misalignment)
- ✅ **Auto-Cleaning**: Stock splits detected and cleaned automatically
- ✅ **Pre-Validation**: Data quality checked before training
- ✅ **Hash Verification**: Detects corruption automatically

#### **CSV Location (correct folder and filenames):**
All CSV data lives in **`ppo/saved_data/csv_data/`**:

| File | Description | Used by agent today |
|------|-------------|---------------------|
| `trading_data_features.csv` | Daily features (indicators) | Yes (primary) |
| `trading_data_raw.csv` | Daily raw OHLCV | Yes (backup) |
| `data_manifest.json` | Daily metadata + hash | Yes |
| `trading_data_1m.csv` | 1-minute OHLCV (yfinance) | Planned (multi-timeframe) |
| `trading_data_5m.csv` | 5-minute OHLCV (yfinance) | Yes (merged into daily when present) |
| `trading_data_30m.csv` | 30-minute OHLCV (yfinance) | Optional (multi-timeframe) |
| `trading_data_1h.csv` | 1-hour OHLCV (yfinance) | Optional (multi-timeframe) |
| `data_manifest_1m.json` … `data_manifest_1h.json` | Per-interval manifests | Optional |

#### **Files Created:**
- `ppo/saved_data/csv_data/trading_data_features.csv` - Main dataset (13,266 rows, 18 columns)
- `ppo/saved_data/csv_data/trading_data_raw.csv` - Raw OHLCV backup
- `ppo/saved_data/csv_data/data_manifest.json` - Metadata with hash
- Intraday: `trading_data_1m.csv`, `trading_data_5m.csv`, `trading_data_30m.csv`, `trading_data_1h.csv` (same folder)

#### **Usage:**
```powershell
# Step 1: Generate CSV (one-time)
python generate_csv_data.py

# Step 1b (optional): Intraday CSV from yfinance (1m, 5m, 30m, 1h)
python generate_intraday_csv.py --intervals 1m 5m 30m 1h
# Output: ppo/saved_data/csv_data/trading_data_1m.csv, trading_data_5m.csv, etc.
# Yahoo limits: 1m = 7 days (or up to 30 via --1m-days); 5m/30m/1h = 59 days (within 60-day limit).

# Step 2: Pre-compute calculations (one-time, speeds up training)
python precompute_calculations.py

# Step 3: Train PPO (uses CSV automatically)
python PPO_fixed.py
```

#### **Benefits:**
- ✅ **Deterministic Training**: Same data every run
- ✅ **No API Calls During Training**: Faster, more reliable
- ✅ **Corruption Prevention**: Pre-validation catches issues
- ✅ **Reproducibility**: Hash verification ensures data integrity

#### **Multi-Timeframe Data Plan (1m, 5m, 30m, 1h, 1d):**
- **Goal**: Feed 1-minute, 5-minute, 30-minute, 1-hour, and 1-day data to the agent; trade depending on market (e.g. regime, volatility).
- **Current**: Agent uses **daily** data from `trading_data_features.csv`. When `trading_data_5m.csv` exists, **intraday aggregates** (intraday_vol, intraday_range_pct, intraday_vol_sum, intraday_vol_ratio) are merged into the daily DataFrame by `CSVDataManager.load_processed_data(merge_intraday=True)` (default), so observations can include intraday insight. Disable with `load_processed_data(merge_intraday=False)`.
- **Planned**: Full multi-timeframe env (multiple interval CSVs in observation); policy or env will choose/weight timeframe based on market (e.g. higher weight 5m/30m/1h in volatile/intraday regimes).

#### **Multi-Timeframe Reward Structure (5m, 30m, 1h favored):**
- **Goal**: Restructure rewards so that when multi-timeframe is enabled, 5m, 30m, and 1h timeframes are favored over 1m and 1d.
- **Config** (in `ppo/utils/dynamic_config.py`): `timeframe_reward_weight_1d = 1.0`, `timeframe_reward_weight_1h = 1.1`, `timeframe_reward_weight_30m = 1.15`, `timeframe_reward_weight_5m = 1.2`, `timeframe_reward_weight_1m = 1.0`. These scale rewards for trades aligned with each timeframe's signals (not wired until multi-timeframe env is implemented).

#### **Adding new market data**
- **New tickers (daily)**  
  - Edit `config.py` → `DATA_CONFIG['tickers']` (list of symbols).  
  - Run `python generate_csv_data.py` to download daily OHLCV and write `ppo/saved_data/csv_data/trading_data_features.csv` and `trading_data_raw.csv`.  
  - **Use all companies from an existing CSV:** run `python generate_csv_data.py --tickers-from-csv ppo/saved_data/csv_data/stock_details_5_years.csv` (uses unique `Company` column; Alpaca 200/min may throttle 491 tickers).  
  - At runtime, `ppo.utils.dynamic_config` → `use_all_data_tickers=True` (default) so the agent uses every ticker present in the loaded CSV, not only `DATA_CONFIG['tickers']`.  
  - If you use precompute: `python precompute_calculations.py` after regenerating CSVs.

- **New tickers (intraday)**  
  - Edit `generate_intraday_csv.py` → `DEFAULT_TICKERS` or pass `--tickers SYM1 SYM2 ...` on the command line.  
  - Run `python generate_intraday_csv.py --intervals 1m 5m 30m 1h` (or subset). Outputs go to `ppo/saved_data/csv_data/trading_data_1m.csv`, etc.

- **New intraday intervals**  
  - Script supports `1m`, `5m`, `30m`, `1h`. To add a new interval you’d extend `generate_intraday_csv.py` (INTERVAL_LIMITS, YF_INTERVALS) and Yahoo’s API limits (e.g. 1m max 7 days, 5m/30m/1h max 59 days).

- **Where data lives**  
  - All CSVs: `ppo/saved_data/csv_data/`. Daily: `trading_data_features.csv`, `trading_data_raw.csv`. Intraday: `trading_data_1m.csv`, `trading_data_5m.csv`, `trading_data_30m.csv`, `trading_data_1h.csv`.

- **After adding data**  
  - Today the agent uses **daily** data only (from CSV/data loader). Intraday files are stored for the multi-timeframe plan; when the data loader and env are updated to multi-timeframe, they will read these CSVs and use `timeframe_reward_weight_*` in config.

---

### **✅ Pre-Computed Calculations Cache (NEW - 2026-01-25)**

**Runtime Calculation Cache** - Pre-computes all calculations to speed up training:

#### **Pre-Computed Calculations:**
- ✅ **VWAP** (Volume-Weighted Average Price) - per ticker, per step
- ✅ **Returns** - per ticker, per step
- ✅ **Volatility** - per ticker, per step (20-day rolling std)
- ✅ **ATR** (Average True Range) - per ticker, per step
- ✅ **Market Regime** - per step (trending/mean_reversion)
- ✅ **Volume Ratios** - per ticker, per step (current / average)
- ✅ **Technical Signals** - RSI, MACD, Bollinger Bands, Volume scores

#### **Performance Impact:**
- ✅ **Eliminates Redundant Calculations**: Calculations done once, not every step
- ✅ **Faster Training**: Environment loads pre-computed values instantly
- ✅ **Consistent Results**: Same calculations every run

#### **Cache Files:**
- `ppo/saved_data/calculation_cache/runtime_calculations_latest.pkl` - Main cache
- `ppo/saved_data/calculation_cache/runtime_calculations_YYYYMMDD_HHMMSS.pkl` - Timestamped backup
- `ppo/saved_data/calculation_cache/runtime_calculations_YYYYMMDD_HHMMSS.json` - Metadata

#### **Cache Contents:**
- VWAP matrix: (1206, 11) - All tickers, all steps
- Returns matrix: (1206, 11) - All tickers, all steps
- Volatility matrix: (1206, 11) - All tickers, all steps
- ATR matrix: (1206, 11) - All tickers, all steps
- Market Regime array: (1206,) - Regime per step
- Volume Ratios matrix: (1206, 11) - All tickers, all steps
- Technical Signals dict: 5 signal types (RSI, MACD, MACD Signal, BB z-score, Volume Score)

---

### **Data Quality & Validation:**

#### **Stock Split Detection & Cleaning:**
- ✅ **Auto-Detection**: Detects stock splits (>50% price drops)
- ✅ **Auto-Cleaning**: Interpolates prices to smooth split events
- ✅ **Logged**: All splits detected and cleaned (informational)

#### **Data Validation:**
- ✅ **Price Sanity**: No zero/negative prices
- ✅ **Data Integrity**: No High < Low
- ✅ **Date Alignment**: All tickers aligned to common dates
- ✅ **Indicator Quality**: No NaN, no Inf in indicators
- ✅ **Corruption Detection**: Extreme price rises flagged (splits are drops, not rises)

---

### **Data Sources:**
- **Primary**: Alpaca API (market data)
- **Secondary**: yfinance (fundamental data fallback)
- **Cache Strategy**: CSV for daily data, pickle cache for quarterly fundamentals

**Alpaca & IEX (free tier):**
- **IEX (Investors Exchange)** is the only feed available without a subscription (~2.5% of US equity volume). Ideal for app testing; precise pricing is not the primary focus.
- **Alpaca does NOT provide company fundamental data** (P/E, revenue, earnings, balance sheet). It provides market data only: bars, quotes, trades. For fundamentals this project uses **yfinance** and/or a pre-built cache (`ppo/collect_fundamentals.py` → `ppo/saved_data/fundamental_cache/`).

---

## 🔍 **SHAP Explainability System**

### **Overview**
The agent includes an integrated **SHAP (SHapley Additive exPlanations)** explainability system that automatically generates reports explaining why the agent made each trading decision. This provides transparency into the agent's decision-making process and helps identify which features drive trading actions.

### **Features:**
- ✅ **Automatic Generation**: SHAP reports generated automatically after each evaluation
- ✅ **Comprehensive Analysis**: Top 20 most important features, action-specific analysis (BUY vs SELL)
- ✅ **Individual Explanations**: JSON files for each trading decision with top 15 influential features
- ✅ **Confidence Scores**: Quantifies how confident the agent was in each decision
- ✅ **Feature Importance**: Shows which technical/fundamental indicators matter most
- ✅ **Performance Optimized**: Limited samples (50) for faster computation

### **Output Files:**
After each training run, check `ppo/explanations/`:
- **`shap_report_YYYYMMDD_HHMMSS.md`** - Comprehensive SHAP analysis report
- **`explanation_step_*_ACTION_TICKER.json`** - Individual decision explanations

### **Report Contents:**
1. **Summary Statistics**: Total decisions explained, features analyzed, SHAP status
2. **Action Distribution**: BUY/SELL/HOLD percentages
3. **Top 20 Most Important Features**: Overall feature importance across all decisions
4. **Feature Importance by Action**: Different features for BUY vs SELL decisions
5. **Confidence Analysis**: Average confidence, high/low confidence breakdown
6. **Key Insights**: Recommendations for feature engineering and model improvement

### **Requirements:**
- **Optional but Recommended**: `pip install shap`
  - If SHAP is not installed, the system will use signal analysis only (no SHAP values)
  - Reports will still be generated with available data

### **Configuration:**
Adjust explanation settings in `PPO_fixed.py` (around line 4115):
```python
shap_results = explainer.explain_evaluation_run(
    env=test_env,
    model=model,
    max_steps=500,              # Max steps to evaluate
    explain_trades_only=True,   # Only explain BUY/SELL (skip HOLD)
    max_explanations=50        # Max number of explanations
)
```

### **Usage:**
The SHAP system runs automatically after each evaluation. No manual intervention required. Simply run:
```bash
python PPO_fixed.py
```

After completion, check `ppo/explanations/shap_report_*.md` for the comprehensive analysis.

### **Benefits:**
- **Transparency**: Understand why the agent makes specific trading decisions
- **Feature Engineering**: Identify which indicators are most influential
- **Model Debugging**: Spot issues with feature importance or action bias
- **Performance Optimization**: Use insights to refine reward structure and features
- **Action Analysis**: Compare feature importance for BUY vs SELL decisions

---

## 📈 **Development Phases**

### **PHASE 1: Foundation Setup** ✅
**Goal**: Get basic PPO agent running
- ✅ Set up PPO training pipeline
- ✅ Implement basic trading environment
- ✅ Configure 12-ticker portfolio (diverse sectors: Tech, Financial, Consumer, Healthcare, Energy, Commodities, Bonds, Crypto)
- ✅ Establish reward structure

**Results**: Agent learned to trade but with suboptimal performance

### **PHASE 2: Penalty Removal & Reward Optimization** ✅
**Goal**: Fix critical issues preventing profitable trading

#### **Key Changes:**
1. **✅ Removed Quick Flip Penalties**
   - Eliminated `-0.5 * (1.0 - hold_time / 10.0)` penalty
   - Agent was being punished for rapid trades

2. **✅ Unified Hold Rewards**
   - Same logic for training and evaluation
   - Removed `eval_mode` check creating different reward structures

3. **✅ Increased PnL Multiplier**
   - Changed from `0.1` to `5.0` (50x stronger profit signals)
   - Profitable trades now get much stronger positive rewards

4. **✅ Removed Failed Sell Penalties**
   - Set `failed_sell_penalty` and `failed_sell_consecutive_penalty` to `0.0`
   - No more punishment for invalid sell attempts

5. **✅ Optimized Learning Parameters**
   - Set `learning_rate` to `1e-05` for BC fine-tuning
   - Reduced `ent_coef` to `0.01` to preserve BC strategy

**Results**: Agent started trading actively (1,310 trades) with 14.64% returns

### **PHASE 3: Realistic Costs & Super Agent Optimization** ✅
**Goal**: Scale to super agent with realistic Alpaca costs

#### **Transaction Cost Optimization:**
1. **✅ Fixed Alpaca Transaction Costs**
   - **Before**: 0.1% per trade (unrealistic)
   - **After**: 0.0024% per trade (real Alpaca fees)
   - **Impact**: 416x lower transaction costs

2. **✅ Reduced Transaction Penalties**
   - **Before**: `transaction_penalty: 1.2`
   - **After**: `transaction_penalty: 0.3` (75% reduction)
   - **Result**: Encourages more trading

#### **Reward Structure Optimization:**
1. **✅ Increased PnL Multipliers**
   - **BUY rewards**: `0.2` → `100.0` (500x stronger)
   - **SELL rewards**: `0.2` → `100.0` (500x stronger)
   - **Base PnL**: `5.0` → `100.0` (20x stronger)

2. **✅ Added VWAP-Based Intelligence**
   - Implemented `_calculate_vwap_reward()` function
   - Rewards buying below VWAP (good entry)
   - Penalizes buying above VWAP
   - Matches BC strategy intelligence

3. **✅ Fixed Variable Names**
   - **Issue**: `NameError: name 'shares_to_buy' is not defined`
   - **Fix**: Changed to `position_size` in BUY section
   - **Result**: No more runtime crashes

#### **Code Quality Improvements:**
1. **✅ Fixed Indentation Errors**
   - Corrected all indentation issues in `trading_env.py`
   - Fixed BUY reward function indentation
   - Fixed VWAP reward function calls
   - Removed duplicate else statements

2. **✅ Validated Python Syntax**
   - All syntax errors resolved
   - Clean, maintainable code structure

### **PHASE 5: Reward Structure Optimization & BC Regularization** ✅
**Goal**: Fix reward structure so signals drive decisions, prevent BC unlearning

#### **Reward Structure Optimization:**
1. **✅ Signals Now Drive 85% of Reward**
   - **Before**: Portfolio change 10% + Technical 20% + Fundamental 15% = 45% signal influence
   - **Current**: RL rewards use technical signals only (technical_signal_weight=1.0, fundamental_signal_weight=0.0); fundamental reserved for supervised networks later
   - **Impact**: Agent learns signals matter more than short-term portfolio moves
   - **Location**: `ppo/models/trading_env.py` lines 2782-2824, 3133-3174

2. **✅ Signal Amplification System**
   - **Strong Signals (0.7-0.8)**: Get 1.5x multiplier
   - **Very Strong Signals (>0.8)**: Get 2.0x multiplier
   - **Impact**: High-quality signals create much higher rewards
   - **Location**: `ppo/models/trading_env.py` lines 2798-2805, 3148-3155

3. **✅ Signal Confirmation Bonuses**
   - **High-Confidence Trade** (tech > 0.7 AND fund > 0.6): +0.3 bonus
   - **Signal Alignment** (both bullish/bearish): +0.2 bonus
   - **Impact**: Rewards agent for following strong, aligned signals
   - **Location**: `ppo/models/trading_env.py` lines 2810-2820, 3160-3170

4. **✅ Enhanced Hold Penalty with Signal Awareness**
   - **Strong Signal Penalty**: If strong signals exist (tech > 0.5 OR fund > 0.6) and agent HOLDS: -0.1 penalty
   - **Progressive Hold Penalty**: Exponential penalty after 20 consecutive holds
   - **Impact**: Agent learns "strong signals = must trade, not hold"
   - **Location**: `ppo/models/trading_env.py` lines 3377-3405

5. **✅ Increased Trade Frequency Threshold**
   - **Before**: Trade frequency penalty after 10 trades
   - **After**: Trade frequency penalty after 50 trades
   - **Impact**: Allows more trading activity before penalties kick in
   - **Location**: `ppo/models/trading_env.py` lines 2870-2876, 3221-3227

6. **✅ Test Mode Consistency**
   - **Hold Penalty**: Increased from -0.001 to -0.01 in test mode (matches training)
   - **Trading Incentive**: Increased from 0.01 to 0.10 in test mode (full incentive)
   - **Impact**: Consistent behavior between training and evaluation
   - **Location**: `ppo/models/trading_env.py` lines 1727-1731, 3365-3371

#### **BC Regularization Implementation (OPTIMIZED):**
1. **✅ BC Regularization Callback (OPTIMIZED)**
   - **Purpose**: Preserves BC knowledge during PPO training (prevents unlearning)
   - **Mechanism**: Monitors policy drift from BC, adjusts learning rate dynamically
   - **OPTIMIZATION**: Checks every 1000 steps (was every step) - **99.9% reduction in computation**
   - **OPTIMIZATION**: Much lighter penalty - 5% LR reduction (was 50%) on high drift - **90% lighter**
   - **OPTIMIZATION**: Regularization weight reduced from 0.3 to 0.03 - **10x lighter**
   - **Caching**: Drift value cached for 1000 steps to avoid redundant calculations
   - **High Drift (>50%)**: Reduces LR by 5% (was 50%) to preserve BC
   - **Moderate Drift**: Reduces LR proportionally with lighter weight
   - **Low Drift (<10%)**: Normal LR
   - **Duration**: Active for first 150,000 timesteps
   - **Location**: `ppo/training/ppo_trainer.py` lines 41-170

2. **✅ BC Policy State Storage**
   - **BC Weights Stored**: After BC initialization, weights are saved for comparison
   - **Regularization Weight**: 0.03 (3% preservation strength, was 0.3)
   - **Integration**: Combined with existing progress callback
   - **Location**: `ppo/training/ppo_trainer.py` lines 357-362

3. **✅ Training Integration**
   - **Combined Callbacks**: BC regularization + progress tracking via CallbackList
   - **Applied to**: Both new training and continued training paths
   - **Location**: `PPO_fixed.py` lines 2224-2230, 2482-2492

#### **Critical Bug Fixes:**
1. **✅ Fixed UnboundLocalError in HOLD Action**
   - **Issue**: `ticker` variable not defined in HOLD action path (line 3380)
   - **Error**: `UnboundLocalError: cannot access local variable 'ticker'`
   - **Fix**: Changed to check all tickers for strong signals (HOLD affects all tickers)
   - **Impact**: No more crashes during HOLD actions
   - **Location**: `ppo/models/trading_env.py` lines 3377-3405

#### **Configuration Updates:**
1. **✅ Signal Weights (current: technical-only for RL)**
   - **dynamic_config**: technical_signal_weight=1.0, fundamental_signal_weight=0.0 (RL rewards technical-only; fundamental reserved for supervised nets later)
   - **Location**: `ppo/utils/dynamic_config.py`

#### **Why This Matters:**
**Before Phase 5:**
- Agent learned to HOLD (97% of actions) because rewards made holding optimal
- BC knowledge was quickly overwritten (unlearned in ~10k steps)
- Agent optimized for avoiding trades, not making good trades

**After Phase 5 / current:**
- RL rewards use technical signals only → agent learns to trade on technical signals
- BC regularization preserves trading knowledge for 50k steps
- Strong signals get amplified rewards → agent learns signal recognition
- Hold penalty increases when signals exist → agent trades instead of holding
- **Result**: Agent learns to trade intelligently based on signals, not avoid trading

---

### **PHASE 4: Technical & Fundamental Analysis Integration** ✅
**Goal**: Teach agent to use technical indicators and fundamental analysis

#### **Technical Analysis Implementation:**
1. **✅ RSI (Relative Strength Index)**
   - **Oversold (<30)**: Strong BUY signal
   - **Overbought (>70)**: Strong SELL signal
   - **Neutral (30-70)**: Moderate signals
   - **Integration**: Weighted into reward calculation

2. **✅ MACD (Moving Average Convergence Divergence)**
   - **Bullish Crossover**: Strong BUY signal
   - **Bearish Crossover**: Strong SELL signal
   - **Above Signal Line**: Moderate BUY
   - **Below Signal Line**: Moderate SELL

3. **✅ Bollinger Bands**
   - **Lower Band Touch**: Strong BUY (oversold)
   - **Upper Band Touch**: Strong SELL (overbought)
   - **Middle Band**: Neutral signal
   - **Band Width**: Volatility measurement

4. **✅ Volume Analysis**
   - **High Volume (>120% avg)**: Confirms signal strength
   - **Low Volume (<80% avg)**: Weakens signal
   - **Volume Confirmation**: Required for strong signals

5. **✅ Market Regime Detection**
   - **ATR-Based Detection**: Trending vs Mean-reversion markets
   - **Trending Markets**: Rewards momentum signals (MACD)
   - **Mean-Reversion Markets**: Rewards contrarian signals (RSI, Bollinger)
   - **Dynamic Adjustment**: Signal weights adapt to market conditions

#### **Fundamental Analysis Implementation:**
1. **✅ Financial Health Scoring**
   - **EXCELLENT (85+)**: Strong buy signal for BUY, hold for SELL
   - **GOOD (70-84)**: Moderate buy signal for BUY, moderate sell for SELL
   - **FAIR (40-69)**: Neutral signal
   - **POOR (<40)**: Avoid signal for BUY, strong sell for SELL

2. **✅ PE Ratio Analysis**
   - **12-20**: Excellent value (strong BUY signal)
   - **20-25**: Good value (moderate BUY signal)
   - **25-30**: Moderate (neutral signal)
   - **>30**: Overvalued (SELL signal)

3. **✅ ROE (Return on Equity)**
   - **>15%**: Excellent profitability (strong BUY)
   - **10-15%**: Good profitability (moderate BUY)
   - **5-10%**: Moderate (neutral)
   - **<5%**: Poor (avoid/SELL)

4. **✅ Debt-to-Equity Ratio**
   - **<0.5**: Low debt, strong financial position (BUY)
   - **0.5-1.0**: Moderate debt (neutral)
   - **>1.0**: High debt, risky (avoid/SELL)

5. **✅ Growth Metrics**
   - **Revenue + Earnings >10%**: Strong growth (BUY)
   - **Revenue + Earnings >5%**: Good growth (moderate BUY)
   - **Negative Growth**: Avoid (SELL)

#### **Reward Shaping System (PHASE 4 - UPDATED IN PHASE 5):**
1. **✅ Phase 4 Reward Formula** (OLD)
   ```python
   # 60% P&L + 20% Technical + 15% Fundamental + 5% Other
   reward = (portfolio_change * 0.60) + 
            (technical_bonus * 0.20) + 
            (fundamental_bonus * 0.15) + 
            (other_bonuses * 0.05)
   ```

2. **✅ Phase 5 / Current Reward Formula**
   ```python
   # RL rewards: technical-only (fundamental_signal_weight=0; fundamental reserved for supervised nets)
   # dynamic_config: technical_signal_weight=1.0, fundamental_signal_weight=0.0
   # Reward uses technical_bonus (tech_w=1.0), PnL, signal_alignment_bonus, other terms from reward_config
   # Signal amplification: strong signals get 1.5x-2.0x multiplier in env
   ```

2. **✅ Technical Signal Integration**
   - **BUY Actions**: Get technical bonus for good RSI/MACD/Bollinger setups
   - **SELL Actions**: Get technical bonus for overbought/divergence signals
   - **Regime Awareness**: Different signals for trending vs mean-reversion

3. **✅ Fundamental Signal Integration**
   - **BUY Actions**: Get fundamental bonus for financially healthy companies
   - **SELL Actions**: Get fundamental bonus for avoiding poor companies
   - **Quality Focus**: Agent learns to prefer quality over speculation

#### **Configuration Updates:**
1. **✅ Technical Signal Config**
   ```python
   TECHNICAL_SIGNAL_CONFIG = {
       'technical_signal_weight': 0.20,
       'rsi_oversold': 30, 'rsi_overbought': 70,
       'bollinger_period': 20, 'bollinger_std': 2.0,
       'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
       'volume_threshold': 1.2,
       'atr_period': 14,
       'high_volatility_threshold': 0.03
   }
   ```

2. **✅ Fundamental Signal Config**
   ```python
   FUNDAMENTAL_SIGNAL_CONFIG = {
       'fundamental_signal_weight': 0.15,
       'fundamental_health_threshold': 40,
       'fundamental_excellent_threshold': 85,
       'pe_ratio_good_min': 12, 'pe_ratio_good_max': 20,
       'roe_strong_threshold': 0.15, 'roe_good_threshold': 0.10,
       'debt_equity_low_threshold': 0.5,
       'growth_strong_threshold': 0.10
   }
   ```

#### **Critical Bug Fixes:**
1. **✅ Fixed NoneType Reward Bug**
   - **Issue**: `_execute_trade()` returning `None` in some code paths
   - **Fix**: Added proper `else` clause for insufficient funds case
   - **Result**: No more `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'float'`

2. **✅ Fixed Indentation Errors**
   - **Issue**: Autopep8 formatter created structural problems
   - **Fix**: Corrected all indentation levels and control flow
   - **Result**: Clean, maintainable code structure

---

### **PHASE 6: Performance Optimization & Advanced Dynamic Stop Loss** ✅ (IN PROGRESS)
**Goal**: Optimize execution speed and implement intelligent stop loss system

#### **Performance Optimizations (10-20x Speed Improvement):**

1. **✅ Optimized Signal Caching**
   - **Before**: Cache cleared and recreated for EVERY action call (~71 calls/step)
   - **After**: Cache persists for entire step, only cleared when timestep changes (~10 calls/step)
   - **Implementation**: Added `_signal_cache_step` tracking, use `dict.clear()` instead of `{}`
   - **Impact**: **~7x fewer** signal calculations per step
   - **Location**: `ppo/models/trading_env.py` lines 2717-2728, 2805-2815

2. **✅ Fast Vectorized Portfolio Calculation**
   - **Before**: Portfolio recalculated 4+ times per step using Python loops
   - **After**: Cached vectorized calculation with NumPy, reused for entire step
   - **Method**: `_calculate_portfolio_value_fast()` with step-level caching
   - **Implementation**: NumPy vectorization + cache check via `_cached_portfolio_step`
   - **Impact**: **~5-10x faster** portfolio calculations
   - **Location**: `ppo/models/trading_env.py` lines 4132-4157

3. **✅ Config Value Caching**
   - **Before**: `get_dynamic_config()` called repeatedly for RSI/MACD thresholds (every signal call)
   - **After**: Config values cached once on first use, reused thereafter
   - **Implementation**: `_cached_ta_config` dictionary stores RSI/MACD/volume thresholds
   - **Impact**: **~50% faster** technical signal calculations, eliminated redundant imports
   - **Location**: `ppo/models/trading_env.py` lines 2731-2742

4. **✅ Reduced Logging Overhead**
   - **Before**: Detailed logs every 5th trade
   - **After**: Detailed logs every 20th trade only
   - **Implementation**: Changed modulo check from `% 5` to `% 20`
   - **Impact**: **~4x faster** I/O operations, cleaner logs
   - **Location**: `ppo/models/trading_env.py` lines 3263, 3168

5. **✅ Debug Logging Gating**
   - **Before**: Debug reward logs printed for every BUY trade
   - **After**: Debug reward logs only every 20th trade
   - **Implementation**: Added conditional check before debug logging
   - **Impact**: **~1.5x faster** trade execution
   - **Location**: `ppo/models/trading_env.py` lines 3167-3169

6. **✅ BC Regularization Optimization (NEW - Phase 7)**
   - **Before**: BC drift checked every step, heavy penalty (50% LR reduction), weight 0.3
   - **After**: BC drift checked every 1000 steps, light penalty (5% LR reduction), weight 0.03
   - **Implementation**: Added caching, reduced check frequency, lighter penalties
   - **Impact**: **99.9% reduction** in BC regularization computation overhead
   - **Location**: `ppo/training/ppo_trainer.py` lines 41-170

7. **✅ Evaluation Frequency Optimization (NEW - Phase 7)**
   - **Before**: Adaptive and early stopping callbacks evaluated every 50k steps
   - **After**: Evaluation frequency increased to 100k steps
   - **Impact**: **50% reduction** in evaluation overhead
   - **Location**: `PPO_fixed.py` lines 2476-2493

8. **✅ Progress Logging Optimization (NEW - Phase 7)**
   - **Before**: Detailed progress logged every 10k steps
   - **After**: Log frequency increased to 25k steps
   - **Impact**: **60% reduction** in I/O overhead
   - **Location**: `PPO_fixed.py` lines 2469, 2992

**Overall Performance Gains:**
- **Steps/Second**: 0.2 → 2-4 (**10-20x faster**)
- **Training Time (250k steps)**: 347 hours → 17-35 hours (**90% reduction**)
- **Signal Calculations**: 71/step → 10/step (**7x fewer**)
- **Portfolio Calculations**: 4+/step → 1 (cached) (**4x fewer**)
- **BC Regularization**: Every step → Every 1000 steps (**99.9% reduction**)
- **Evaluation Overhead**: 50k → 100k frequency (**50% reduction**)
- **Progress Logging**: 10k → 25k frequency (**60% reduction**)

#### **Advanced Dynamic Stop Loss System:**

1. **✅ Volatility-Adjusted Stop Loss**
   - **Base Stop Loss**: 10% (configurable via `base_stop_loss_pct` in dynamic_config)
   - **20-Day Volatility Adjustment**: Widens stop loss in volatile markets
   - **Formula**: `vol_adjustment = (recent_volatility / 0.02) * 0.02` (max +2%)
   - **Purpose**: Prevents premature exits during normal market fluctuations
   - **Location**: `ppo/models/trading_env.py` lines 4205-4212

2. **✅ ATR (Average True Range) Integration**
   - **ATR Calculation**: 14-period ATR from price matrix
   - **ATR Adjustment**: `(atr / current_price) * 0.5` (max +3%)
   - **Purpose**: Adapts stop loss to instrument-specific volatility
   - **Impact**: Different stop losses for different stocks based on their trading patterns
   - **Location**: `ppo/models/trading_env.py` lines 4214-4223

3. **✅ Market Regime Awareness**
   - **Bull Market**: Tighter stop loss (-1%, allow profits to run)
   - **Bear Market**: Wider stop loss (+2%, avoid shakeouts)
   - **Volatile Market**: Significantly wider stop loss (+4%, survive noise)
   - **Default (Unknown)**: No adjustment
   - **Purpose**: Adapts to market conditions automatically
   - **Location**: `ppo/models/trading_env.py` lines 4225-4233

4. **✅ Signal-Aware Stop Loss**
   - **Strong Negative Signals** (tech < 0.3): Tighter stop loss (-2%, exit faster)
   - **Weak/Mixed Signals** (0.3-0.7): No adjustment (standard stop)
   - **Strong Positive Signals** (tech > 0.7): Wider stop loss (+1%, let winners run)
   - **Purpose**: Uses technical signals to predict potential drops
   - **Location**: `ppo/models/trading_env.py` lines 4235-4242

5. **✅ Signal-Aware Stop Loss Rewards**
   - **Expected Drop** (signals warned, tech < 0.3): **-0.1 penalty** (small, agent was warned)
   - **Unexpected Drop** (no warning, tech > 0.3): **-0.5 penalty** (full, agent should've seen it)
   - **Trailing Stop (profit-taking)**: **-0.05 penalty** (very small, encouraged behavior)
   - **Purpose**: Teaches agent to respect technical signals predicting drops
   - **Impact**: Agent learns to exit BEFORE stop loss triggers when signals warn
   - **Location**: `ppo/models/trading_env.py` lines 4248-4260

**Complete Dynamic Stop Loss Formula:**
```python
dynamic_stop_loss = base_stop_loss (10%)
                  + volatility_adjustment (0-2%)
                  + atr_adjustment (0-3%)
                  + regime_adjustment (-1% to +4%)
                  + signal_adjustment (-2% to +1%)
# Final range: 5% to 30% (bounded for safety)
```

#### **Signal-Confidence Gating System:**

1. **✅ Confidence Calculation**
   - **Formula**: `confidence = (|tech_signal| + |fund_signal|) / 2.0` (or tech only when fundamentals disabled)
   - **Range**: 0.0 (no confidence) to 1.0 (maximum confidence)
   - **Purpose**: Measures how strong and aligned signals are
   - **Location**: `ppo/models/trading_env.py` _should_gate_trade_by_signal_confidence

2. **✅ Signal Alignment Check**
   - **BUY Alignment**: Both tech > 0.5 AND fund > 0.5 (bullish); when fundamentals disabled, tech > 0
   - **SELL Alignment**: Tech < 0.5 OR fund < 0.5 (bearish/caution); when fundamentals disabled, aligned = True
   - **Location**: `ppo/models/trading_env.py` _should_gate_trade_by_signal_confidence

3. **✅ Three-Tier Gating (hardcoded in env)**
   - **High confidence (>0.25 + aligned)**: Trade: YES, Reward multiplier: **1.0x**
   - **Medium confidence (>0.10 + aligned)**: Trade: YES, Reward multiplier: **0.7x**
   - **Low confidence**: Trades allowed only under cooldown (e.g. every 10 steps) when signals aligned; reward multiplier **0.3x**. Very low (<0.05) or conflicting signals: block. Not a fixed 20% random exploration.
   - **Purpose**: Quality over quantity; thresholds are hardcoded in _should_gate_trade_by_signal_confidence (config defines defaults but env does not read them in that function).

4. **✅ Exploration Mechanism**
   - Low-confidence path is cooldown-based (allow trade every 10 steps when aligned), not a fixed % random.

#### **BUY/SELL Reward Rebalancing:**

1. **✅ Base Incentives (from dynamic_config)**
   - **BUY**: base_trading_incentive 0.50 (scales 0.7x-1.6x with market)
   - **SELL**: sell_base_incentive 0.48, exploration_bonus 0.50 (same scaling)
   - **Purpose**: Prevents BUY-bias, encourages strategic selling
   - **Location**: `ppo/models/trading_env.py` _execute_trade reward paths; reward_config synced from dynamic_config

2. **✅ Consistent Signal-Based Bonuses**
   - **Signal Alignment Bonus**: Up to +0.3 for high-confidence aligned signals (BOTH). **2026-02-09**: For SELL, full bonus only when **realized PnL > 0**; when PnL ≤ 0 bonus = 0; 0 < PnL ≤ 0.001 at 25%.
   - **Learning Bonus**: +0.05 for early timesteps (BOTH)
   - **Profitable Trade Bonus**: Scaled by PnL % (BOTH)
   - **First Trade Bonus**: +0.1 for first trade of episode (BUY only)
   - **Purpose**: Rewards quality trades regardless of direction
   - **Location**: `ppo/models/trading_env.py` lines 3124-3157, 3494-3533

3. **✅ Deprecation of Old Parameters**
   - **Deprecated**: `trading_incentive` (was BUY-biased)
   - **Replaced by**: `base_trading_incentive` (direction-neutral)
   - **Migration**: Config updated, backward compatibility maintained
   - **Location**: `ppo/utils/dynamic_config.py` lines 105-110

#### **Configuration Updates:**

1. **✅ New Reward Parameters**
   ```python
   # Reward rebalancing (Phase 6)
   base_trading_incentive: 0.05      # NEW: Equal base for BUY/SELL
   signal_alignment_bonus: 0.3       # NEW: High-confidence aligned signals
   profitable_trade_bonus: 0.2       # NEW: Scaled by PnL %
   learning_bonus: 0.05              # NEW: Early timesteps bonus
   first_trade_bonus: 0.1            # NEW: First trade of episode
   # trading_incentive: 0.05         # DEPRECATED: Use base_trading_incentive
   ```
   - **Location**: `ppo/utils/dynamic_config.py` lines 105-110

2. **✅ Progressive Scaling Monitoring**
   ```python
   # Scaling parameters (Phase 6)
   total_timesteps: 250000           # CENTRALIZED: Current phase
   scaling_phase: "250k"             # Current phase (dynamic_config)
   min_sharpe_to_scale: 0.2          # Min Sharpe to increase timesteps
   min_return_to_scale: 0.02         # Min return (2%) to scale up
   max_drawdown_to_scale: 0.15       # NEW: Max drawdown to allow scaling
   timestep_increase_amount: 50000   # NEW: How much to increase
   timestep_decrease_amount: 50000   # NEW: How much to decrease on errors
   ```
   - **Location**: `ppo/utils/dynamic_config.py` lines 162-183

#### **Why Phase 6 Matters:**

**Before Phase 6:**
- Training was SLOW: 5 seconds/step = 347 hours for 250k steps
- Simple stop loss: Fixed 10% base (config) threshold, now adapts via volatility + ATR + regime + signals
- BUY-bias: Different reward structures for BUY vs SELL
- No quality control: Agent traded on weak signals

**After Phase 6:**
- Training is FAST: 0.25-0.5 seconds/step = 17-35 hours for 250k steps (**10-20x faster**)
- Advanced stop loss: Adapts to volatility, ATR, regime, and signals
- Signal-aware penalties: Agent learns to respect technical warnings
- Balanced rewards: BUY and SELL have equal base incentives
- Quality gating: Only high-confidence signals get full rewards
- **Result**: Faster training + smarter stop loss + balanced trading + quality focus = **SUPER AGENT**

### **PHASE 7: Adaptive Hyperparameter Optimization & Architecture Upgrade** ✅ (IN PROGRESS)
**Goal**: Prevent over-conservatism at high timesteps, maintain optimal trading activity, upgrade to TCN temporal memory

#### **TCN Memory Architecture Upgrade (NEW):**
1. **✅ TCN (Temporal Convolutional Network) Implementation**
   - **Upgrade**: Changed from standard PPO (MlpPolicy) to PPO with TCN policy (TCNTradingPolicy)
   - **TCN Configuration**: hidden_dims=[256, 256, 128, 128], dilation_rates=[1, 2, 4, 8], kernel_size=3
   - **Benefits**: Better long-range memory than LSTM (15+ steps vs 10), faster on CPU, parallelizable
   - **Fallback**: Automatically falls back to standard MlpPolicy if TCN unavailable
   - **Model Compatibility**: Old PPO models compatible (TCN works with standard PPO)
   - **Location**: `ppo/models/tcn_policy.py`, `ppo/training/ppo_trainer.py` lines 605-627

2. **✅ TCN Architecture Details**
   - **Dilated Convolutions**: Exponential dilation [1, 2, 4, 8] provides effective memory of 15+ steps
   - **Residual Connections**: Prevent vanishing gradients, enable deeper networks
   - **Parallelizable**: All convolutions run in parallel (2x faster than LSTM on CPU)
   - **Long-range Memory**: Better than LSTM for capturing long-term temporal dependencies
   - **Location**: `ppo/models/tcn_policy.py`

#### **Training Efficiency Optimizations (NEW):**
1. **✅ BC Regularization Optimization**
   - **Check Frequency**: Every 1000 steps (was every step) - **99.9% reduction**
   - **Penalty Weight**: Reduced from 0.3 to 0.03 - **10x lighter**
   - **LR Reduction**: 5% on high drift (was 50%) - **90% lighter penalty**
   - **Caching**: Drift value cached for 1000 steps to avoid redundant calculations
   - **Impact**: Massive reduction in callback overhead while maintaining BC preservation
   - **Location**: `ppo/training/ppo_trainer.py` lines 41-170

2. **✅ Evaluation Frequency Optimization**
   - **Adaptive Callback**: Increased from 50k to 100k steps
   - **Early Stopping**: Increased from 50k to 100k steps
   - **Impact**: **50% reduction** in evaluation overhead
   - **Location**: `PPO_fixed.py` lines 2476-2493

3. **✅ Progress Logging Optimization**
   - **Log Frequency**: Increased from 10k to 25k steps
   - **Impact**: **60% reduction** in I/O overhead
   - **Location**: `PPO_fixed.py` lines 2469, 2992

#### **Adaptive Learning Rate System:**
1. **✅ Performance-Based LR Adjustment**
   - **Trigger**: Evaluates every 100K timesteps (OPTIMIZED: was 50K)
   - **Condition**: If Sharpe > 2.0 AND Return > 3%
   - **Action**: Increases LR floor by 1.5x (configurable via `adaptive_lr_boost_factor`)
   - **Reset**: Automatically resets to original when performance drops
   - **Purpose**: Prevents over-conservatism when agent is performing well
   - **Location**: `PPO_fixed.py` lines 187-265 (AdaptiveHyperparameterCallback)

2. **✅ Dynamic LR Schedule**
   - **Implementation**: Real-time config reading during training
   - **Mechanism**: Schedule function reads `lr_final` from config dynamically
   - **Benefit**: LR adjustments take effect immediately without restarting training
   - **Location**: `PPO_fixed.py` lines 1648-1676 (adaptive_lr_schedule function)

#### **Adaptive Entropy Coefficient System:**
1. **✅ Trade Count-Based Entropy Adjustment**
   - **Trigger**: Evaluates every 100K timesteps (OPTIMIZED: was 50K)
   - **Condition**: If total trades < 50 (configurable via `adaptive_ent_min_trades`)
   - **Action**: Doubles entropy coefficient (0.01 → 0.02) to encourage exploration
   - **Reset**: Automatically resets when trade count recovers
   - **Purpose**: Prevents agent from becoming too conservative and avoiding trades
   - **Location**: `PPO_fixed.py` lines 240-264 (AdaptiveHyperparameterCallback)

#### **Dynamic Market-Condition Reward System:**
1. **✅ Market Condition Multiplier**
   - **Method**: `_calculate_market_condition_multiplier()` calculates dynamic multiplier (0.7x-1.6x)
   - **Factors**:
     - **Volatility** (0.8x-1.5x): High volatility = higher rewards (more opportunity)
     - **Bull/Bear regime** (explicit): `_detect_bull_bear_regime()` uses SPY over lookback; **Bull** → BUY reward boost (`bull_buy_reward_boost`), **Bear** → SELL reward boost (`bear_sell_reward_boost`); SELL in bull uses `sell_bull_market_boost`
     - **Market Trend** (continuous): Gentle tilt by SPY return (BUY/SELL multipliers)
     - **Regime** (1.0x-1.2x): Trending markets = higher rewards than mean-reversion
   - **Applied To**: Both `base_trading_incentive` (0.50 base) and `exploration_bonus` (0.50 base) from dynamic_config
   - **Purpose**: Rewards adapt to market conditions - higher in volatile/trending markets; **dynamic reward insight by bull/bear regime**
   - **Location**: `ppo/models/trading_env.py` (_detect_bull_bear_regime, _calculate_market_condition_multiplier)

1b. **✅ Bull/Bear regime config** (`ppo/utils/dynamic_config.py`)
   - **bull_buy_reward_boost**: 0.06 (extra multiplier for BUY in bull)
   - **bear_sell_reward_boost**: 0.08 (extra multiplier for SELL in bear)
   - **bull_bear_lookback_days**: 10 (SPY lookback for regime)
   - **bull_threshold_pct**: 0.02 (SPY return ≥ 2% = bull)
   - **bear_threshold_pct**: -0.02 (SPY return ≤ -2% = bear)

2. **✅ Reward Config Sync**
   - **Issue**: base_trading_incentive, sell_base_incentive, exploration_bonus from dynamic_config must sync to env reward_config
   - **Fix**: Reward config overlay mapping in trading_env; _validate_and_fix_reward_config_sync
   - **Result**: Base rewards sync from config (e.g. BUY 0.50, SELL 0.48, exploration 0.50) and scale with market
   - **Location**: `ppo/models/trading_env.py` reward_config sync from dynamic_config

3. **✅ Reward Balance (current config)**
   - **Base Trading Incentive**: 0.50 (BUY)
   - **Sell Base Incentive**: 0.48, Exploration Bonus: 0.50
   - **Eval Hold Penalty**: -0.04
   - **Position Incentive**: 0.02
   - **Result**: BUY/SELL clearly better than HOLD

4. **✅ Signal Gating (env behavior)**
   - **High threshold**: 0.25 (hardcoded in _should_gate_trade_by_signal_confidence)
   - **Medium threshold**: 0.10 (hardcoded)
   - **Low confidence**: Cooldown-based (allow trade every 10 steps when aligned), not fixed % random
   - **Result**: More trades allowed while maintaining quality focus

5. **✅ SELL Reward Enhancement (current config)**
   - **BUY Base Incentive**: 0.50 (dynamic_config)
   - **SELL Base Incentive**: 0.48, sell_profit_taking_bonus 0.50, sell_bull_market_boost 0.08
   - **Balanced Loss Penalties**: buy_loss_penalty_cap and sell_loss_penalty_cap both -1.0
   - **Gentle Market Multipliers**: Bull/Bear adjustments gentle (5-10% tilt)
   - **Result**: SELL rewards competitive with BUY; balanced loss penalties; profit-taking encouraged
   - **Location**: `ppo/utils/dynamic_config.py`; `ppo/models/trading_env.py` _execute_trade reward paths

#### **Problem Solved:**
- **Issue**: At 300K timesteps, LR decayed too low (5e-6) → Agent became over-conservative → Trade count dropped 66% (103 → 35) → Return dropped 78% (6.84% → 1.45%)
- **Root Cause**: Linear LR decay regardless of performance, fixed entropy coefficient, **static rewards regardless of market conditions**
- **Solution**: Adaptive LR maintains higher LR when performance is excellent, adaptive entropy encourages trading when count is low, **dynamic rewards adapt to market conditions**
- **Expected Impact**: Better balance between risk management and trading activity throughout long training runs, **rewards automatically adjust to market opportunities**

#### **Configuration (snapshot; canonical: dynamic_config.py):**
```python
# Adaptive hyperparameter thresholds (dynamic_config.py)
adaptive_lr_sharpe_threshold: 2.0      # Sharpe threshold for LR boost (good performance)
adaptive_lr_return_threshold: 0.03      # Return threshold (3%)
adaptive_lr_boost_factor: 1.5
adaptive_lr_poor_sharpe_threshold: 1.5  # Boost LR when performance poor
adaptive_lr_poor_return_threshold: 0.03
adaptive_lr_poor_boost_factor: 2.0
adaptive_lr_poor_max: 5e-4             # 2026-02-09: Must be > typical current LR so poor branch increases LR (was 5e-5/1e-4)
adaptive_ent_min_trades: 50
adaptive_ent_boost_factor: 2.0
adaptive_ent_max: 0.04                 # Maximum entropy cap

# Dynamic Reward System (dynamic_config.py)
base_trading_incentive: 0.50            # BUY base (scales 0.7x-1.6x with market)
sell_base_incentive: 0.48               # SELL base (scales 0.7x-1.6x with market)
exploration_bonus: 0.50                 # Exploration bonus (scales with market)
eval_hold_penalty: -0.04                # Hold penalty in evaluation
position_incentive: 0.02                # Incentive for maintaining positions
low_confidence_exploration: 0.95        # Config; env uses cooldown-based low-confidence gating

# SELL-Specific Rewards (dynamic_config.py)
sell_profit_taking_bonus: 0.50          # Bonus for profitable SELL (scaled by profit)
sell_loss_penalty_cap: -1.0             # Balanced with buy_loss_penalty_cap
buy_loss_penalty_cap: -1.0              # Balanced with SELL
sell_bull_market_boost: 0.08            # Bonus for SELL in bull markets

# Bull/Bear dynamic reward scaling (dynamic_config.py)
bull_buy_reward_boost: 0.06             # Extra BUY multiplier in bull regime
bear_sell_reward_boost: 0.08            # Extra SELL multiplier in bear regime
bull_bear_lookback_days: 10             # SPY lookback for bull/bear detection
bull_threshold_pct: 0.02                # SPY return >= 2% = bull
bear_threshold_pct: -0.02               # SPY return <= -2% = bear
```

#### **Why Phase 7 Matters:**
**Before Phase 7:**
- LR decayed linearly regardless of performance → Over-conservatism at high timesteps
- Entropy fixed at 0.01 → Agent could become too risk-averse
- Trade count could collapse without intervention → Lower returns
- Manual intervention required to adjust hyperparameters

**After Phase 7:**
- LR adapts to performance (bidirectional: boost when good, boost when poor up to 5e-5)
- Entropy adapts to trade count (cap 0.04) when trades < 50
- **Rewards adapt to market conditions** → Base rewards (0.50/0.48/0.50) scale 0.7x-1.6x
- **Base rewards** → Sync from config (base_trading_incentive 0.50, sell_base_incentive 0.48, exploration_bonus 0.50)
- **SELL rewards** → sell_profit_taking_bonus 0.50, sell_bull_market_boost 0.08; loss penalties -1.0 (both BUY and SELL)
- Self-correcting system → Automatically resets when conditions normalize
- **Result**: Agent maintains optimal balance; SELL rewards encourage profit-taking while balanced loss penalties prevent over-punishment.

---

## 🎯 **Performance Metrics**

### **Current Performance (Updated 2026-02-05):**
| Metric | 50k run | 100k run | 150k run (21-34-38) | Status |
|--------|---------|----------|----------------------|--------|
| **Total Return** | -0.24% | -0.22% | 0.05% | ⚠️ Low (slightly positive at 150k) |
| **Sharpe Ratio** | 0.000 | -0.248 | 0.04 | ⚠️ Needs improvement |
| **Win Rate** | 38.6% | 40.9% | 46.2% | ⚠️ Below 50% |
| **Total Trades** | 430 | 473 | 447 | ✅ Active trading |
| **BUY blocked (gating)** | — | — | 41.1% | ⚠️ High block rate |
| **Raw SELL (policy)** | 75.3% | 31.2% | — | 50k: strong SELL bias; 100k: more balanced |
| **Raw HOLD (policy)** | 9.7% | 49.2% | — | 100k: policy holds more |

**Analysis 2026-02-05:** Return stays low (~0.05–0.06%) because reward structure favors **activity and signal alignment** over **realized PnL**; no strong "hold winner" bonus; 41% BUY blocked by signal gating. See **LOW_RETURN_ANALYSIS_AND_PLAN_2026-02-05.md** for root-cause analysis and phased improvement plan. **REWARD_STRUCTURE_ANALYSIS_2026-02-05.md** documents every reward component; **ANALYSIS_AND_IMPROVEMENT_PLAN_2026-02-05.md** for SELL-bias fix and reward rebalance.

### **Legacy Performance (Historical):**
| Metric | Value | Status |
|--------|-------|--------|
| **Total Return** | 14.64% | ✅ Profitable (earlier phase) |
| **Sharpe Ratio** | 0.17 | ⚠️ Needs improvement |
| **Win Rate** | 76.9% | ✅ Excellent |
| **Max Drawdown** | 18.6% | ⚠️ High risk |
| **Total Trades** | 1,310 | ✅ Active trading |
| **Trades/Episode** | 131 | ✅ Good frequency |

### **Target Performance (Super Agent):**
| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| **Return** | >30% | 14.64% | +15.36% |
| **Sharpe** | >1.0 | 0.17 | +0.83 |
| **Drawdown** | <10% | 18.6% | -8.6% |
| **Paper Trading Ready** | YES | NO | Needs work |

---

## 🚀 **Next Steps: PHASE 7 - Validation & Scaling to 1M Timesteps**

### **PHASE 7: Adaptive Hyperparameter Optimization** ✅ (LATEST UPDATE)

#### **Adaptive Learning Rate System:**
1. **✅ Performance-Based LR Adjustment**
   - **Trigger**: Evaluates every 50K timesteps
   - **Condition**: If Sharpe > 2.0 AND Return > 3%
   - **Action**: Increases LR floor by 1.5x (configurable boost factor)
   - **Reset**: Automatically resets to original when performance drops
   - **Purpose**: Prevents over-conservatism when agent is performing well
   - **Location**: `PPO_fixed.py` lines 187-265 (AdaptiveHyperparameterCallback)

2. **✅ Dynamic LR Schedule**
   - **Implementation**: Real-time config reading during training
   - **Mechanism**: Schedule function reads `lr_final` from config dynamically
   - **Benefit**: LR adjustments take effect immediately without restarting training
   - **Location**: `PPO_fixed.py` lines 1648-1676 (adaptive_lr_schedule function)

3. **✅ Configurable Thresholds**
   ```python
   adaptive_lr_sharpe_threshold: 2.0      # Sharpe threshold for LR boost
   adaptive_lr_return_threshold: 0.03     # Return threshold (3%)
   adaptive_lr_boost_factor: 1.5         # LR boost multiplier
   ```
   - **Location**: `ppo/utils/dynamic_config.py` lines 107-110

#### **Adaptive Entropy Coefficient System:**
1. **✅ Trade Count-Based Entropy Adjustment**
   - **Trigger**: Evaluates every 50K timesteps
   - **Condition**: If total trades < 50 (configurable threshold)
   - **Action**: Doubles entropy coefficient (0.01 → 0.02) to encourage exploration
   - **Reset**: Automatically resets when trade count recovers
   - **Purpose**: Prevents agent from becoming too conservative and avoiding trades
   - **Location**: `PPO_fixed.py` lines 240-264 (AdaptiveHyperparameterCallback)

2. **✅ Model & Config Synchronization**
   - **Updates**: Both model's `ent_coef` attribute and config file
   - **Benefit**: Changes persist and are logged for monitoring
   - **Safety**: Capped at maximum (0.02) to prevent excessive exploration

3. **✅ Configurable Thresholds**
   ```python
   adaptive_ent_min_trades: 50            # Minimum trades threshold
   adaptive_ent_boost_factor: 2.0        # Entropy boost multiplier
   adaptive_ent_max: 0.02                 # Maximum entropy cap
   ```
   - **Location**: `ppo/utils/dynamic_config.py` lines 111-113

#### **Integration & Benefits:**
1. **✅ Automatic Callback Integration**
   - **Combined**: Adaptive callback added to CallbackList with BC regularization and early stopping
   - **Frequency**: Evaluates every 100K timesteps (OPTIMIZED: was 50K, same as early stopping)
   - **Location**: `PPO_fixed.py` lines 2422-2432

2. **✅ Expected Behavior at 300K+ Timesteps**
   - **Scenario 1**: Sharpe > 2.0, Return > 3% → LR increases → More adaptability maintained
   - **Scenario 2**: Trades < 50 → Entropy doubles → More exploration → More trades
   - **Scenario 3**: Both conditions normalize → Automatic reset to original values
   - **Result**: Self-correcting system that prevents over-conservatism and low trade counts

3. **✅ Problem Solved**
   - **Issue**: At 300K timesteps, LR decayed too low (5e-6) → Agent became over-conservative → Trade count dropped 66% (103 → 35) → Return dropped 78% (6.84% → 1.45%)
   - **Solution**: Adaptive LR maintains higher LR when performance is excellent, adaptive entropy encourages trading when count is low
   - **Expected Impact**: Better balance between risk management and trading activity

#### **Why This Matters:**
**Before Adaptive Hyperparameters:**
- LR decayed linearly regardless of performance → Over-conservatism at high timesteps
- Entropy fixed at 0.01 → Agent could become too risk-averse
- Trade count could collapse without intervention → Lower returns
- Manual intervention required to adjust hyperparameters

**After Adaptive Hyperparameters:**
- LR adapts to performance → Maintains adaptability when agent is doing well
- Entropy adapts to trade count → Encourages exploration when needed
- Self-correcting system → Automatically resets when conditions normalize
- **Result**: Agent maintains optimal balance between risk management and trading activity throughout long training runs

### **Immediate Actions (Phase 6 Complete - Phase 7 Starting):**
1. **Train to 250k with New Optimizations**
   - ✅ Performance optimized (10-20x faster)
   - ✅ Advanced stop loss implemented
   - ✅ Signal gating active
   - ✅ Adaptive hyperparameters enabled (LATEST)
   - ⏳ Run full 250k training (expect 17-35 hours, was 347 hours)
   - **Recommended**: Continue from 150k checkpoint (preserves learning)

2. **Monitor Performance Gains**
   - Target: **>2 steps/second** (was 0.2)
   - Verify stop loss triggers are adapting to market conditions
   - Check signal confidence distribution (should favor high-confidence trades)
   - Ensure BUY/SELL balance (no more BUY-bias)

3. **Validate Improvements**
   - Compare 250k results vs 150k baseline
   - Look for: Better Sharpe, lower drawdown, quality trades
   - Monitor stop loss effectiveness (fewer unexpected losses)
   - Check reward logs for balanced BUY/SELL incentives

4. **Scale to 1M Timesteps** (Progressive)
   - Phase 6: 250k (current) ✅
   - Phase 7a: 400k (if Sharpe >0.5, Return >10%, Drawdown <15%)
   - Phase 7b: 600k (if Phase 7a meets targets)
   - Phase 7c: 800k (if Phase 7b shows stability)
   - Phase 7d: 1M (final goal before paper trading)
   - **Criteria**: Auto-scale based on metrics in `dynamic_config.py`

### **Long-term Goals:**
- **Phase 7**: Reach 1M timesteps with stable performance (>30% returns, Sharpe >1.0)
- **Phase 8**: Paper Trading Ready (TG5 integration, live market simulation)
- **Phase 9**: Production Deployment (Real money trading with risk management)

---

## 🛠️ **Installation & Usage**

### **Prerequisites:**
```bash
pip install stable-baselines3
pip install torch
pip install numpy pandas
pip install gym
```

### **Running the Agent:**

#### **1. Train BC Model (Prerequisite):**
```bash
python run_bc.py
```

#### **2. Train PPO Agent:**
```bash
python PPO_fixed.py
```

#### **3. Monitor Performance:**
- Check `ppo/daily_reports/` for performance reports
- Review `ppo/saved_data/` for trade history
- Monitor console output for real-time metrics

#### **4. Run tests (bull/bear rewards & intraday merge):**
```bash
# From project root (e.g. C:\Users\Faiaz)
python -m ppo.tests.test_bull_bear_and_intraday
# Or: python ppo/tests/test_bull_bear_and_intraday.py
```
- Verifies `_detect_bull_bear_regime()` (bull/bear/neutral) and `_calculate_market_condition_multiplier()` (BUY/SELL scaling).
- Verifies `CSVDataManager.load_processed_data(merge_intraday=True/False)` and optional intraday aggregate merge when `trading_data_5m.csv` exists.

---

## 📁 **File Structure & System Architecture**

**Root (C:\\Users\\Faiaz):**
```
├── PPO_fixed.py                 # Main orchestrator (train + eval; deterministic = paper-trading)
├── run_bc.py                    # Behavioral Cloning init (run before PPO when rotating training window)
├── config.py                    # DATA_CONFIG, tickers, API settings
├── generate_csv_data.py         # One-time daily CSV pipeline
├── generate_intraday_csv.py     # Intraday CSVs (1m, 5m, 30m, 1h) → ppo/saved_data/csv_data/
├── precompute_calculations.py   # Pre-compute runtime cache for training
├── README.md                    # This file
└── requirements.txt
```

**ppo/ (C:\\Users\\Faiaz\\ppo) – full layout:**
```
ppo/
├── models/
│   ├── trading_env.py           # FixedTradingEnv: rewards, gating, regime, stop-loss
│   ├── tcn_policy.py            # TCN (Temporal Convolutional Network) policy
│   ├── env_factory.py
│   ├── risk_manager.py
│   ├── reward_calculator.py
│   └── regime_manager.py
├── utils/
│   ├── dynamic_config.py        # Canonical config (reward, scaling, gating defaults)
│   ├── regime_controller.py      # Market-first RegimeController (BUY/SELL/HOLD mults, gating thresholds)
│   ├── agent_explainability.py  # SHAP explainability
│   ├── data_validator.py
│   └── ... (config_manager, advanced_metrics, etc.)
├── evaluation/
│   ├── performance_evaluator.py # evaluate_model(); deterministic + stochastic; writes validation artifacts
│   └── unified_report_generator.py # UNIFIED_REPORT_*.md / .json; collapse diagnostics
├── training/
│   ├── ppo_trainer.py           # PPO agent creation (TCN/Mlp), BC regularization callback
│   ├── behavior_cloning.py      # BC dataset + train
│   └── scaling_controller.py
├── data/
│   ├── csv_loader_adapter.py    # CSV → feature matrices (intraday/daily)
│   ├── csv_data_manager.py      # load_processed_data, merge_intraday
│   ├── feature_engineer.py
│   ├── technical_indicators.py
│   └── expert_dataset.py        # BC expert dataset build
├── saved_data/
│   ├── csv_data/                # trading_data_features.csv, trading_data_5m.csv, etc.
│   ├── calculation_cache/       # runtime_calculations_latest.pkl
│   ├── validation/              # trade_history.csv, backtest_results.json, gating_stats.json (deterministic)
│   ├── validation/stochastic_eval/ # Stochastic diagnostic artifacts
│   └── fundamental_cache/
├── saved_models/                # ppo_trading_agent.zip, ppo_bc_init.pt
├── daily_reports/               # UNIFIED_REPORT_*.md, unified_report_*.json
├── explanations/                # shap_report_*.md, explanation_step_*_ACTION_TICKER.json
├── docs/                        # REWARD_CHANGELOG.md, PERFORMANCE_REFINEMENTS_2026-02-09.md, etc.
└── tests/                       # test_quant_improvements_2026_02_09, test_bull_bear_and_intraday, etc.
```

**Data flow (high level):** CSV/data loader → FixedTradingEnv → PPO (TCN) train → evaluate_model( deterministic=True ) → validation artifacts → UnifiedReportGenerator. RegimeController feeds env via `_get_regime_outputs()` (hold_penalty_mult, gate thresholds); behavior-adaptive layer adjusts rewards as safety.

---

## 🔧 **Configuration**

**Canonical config:** `ppo/utils/dynamic_config.py`. README snapshots may lag; prefer the file for current values.

### **Portfolio Tickers (11 Assets - Diverse Market Conditions):**

**Bull Market Performers:**
- **AAPL, MSFT, NVDA** - Technology (cyclical, growth stocks)
- **AMZN** - Consumer Discretionary (e-commerce, cyclical)
- **XLE** - Energy ETF (inflation hedge, recovery play)

**Bear Market Performers (Defensive):**
- **GLD** - Gold ETF (safe haven, store of value)
- **TLT** - 20+ Year Treasury Bonds (inverse correlation, safe haven)
- **KO** - Coca-Cola (Consumer Staples, recession-resistant)
- **JNJ** - Johnson & Johnson (Healthcare, defensive)

**Cyclical/Neutral:**
- **SPY** - S&P 500 Index (market benchmark)
- **JPM** - JPMorgan Chase (Financial, cyclical)

**Total**: 11 tickers (12 in design; 11 in use — BTC-USD not on Alpaca). Covers Technology, Financial, Consumer Staples, Consumer Discretionary, Healthcare, Energy, Commodities, and Bonds.

### **Key Parameters:**
Snapshot from `ppo/utils/dynamic_config.py` (canonical source). As of audit: Feb 2026.

```python
# Transaction Costs (Real Alpaca)
transaction_cost: 0.000024      # 0.0024% per trade
commission_rate: 0.000024       # 0.0024% per trade

# RL Reward Signal Weights (technical-only; fundamental reserved for supervised nets)
technical_signal_weight: 1.0    # 100% technical for RL rewards
fundamental_signal_weight: 0.0  # 0% fundamental for RL (reserved for supervised networks later)
pnl_weight: 0.6                 # Weight for P&L signals (60%)

# Technical / Fundamental (for reward computation; config.py / dynamic_config)
rsi_oversold: 40, rsi_overbought: 60
bollinger_period: 20, bollinger_std: 2.0
macd_fast: 12, macd_slow: 26, macd_signal: 9
volume_threshold: 1.0
atr_period: 14
pe_ratio_good_min: 10.0, pe_ratio_good_max: 20.0
roe_strong_threshold: 0.15, roe_good_threshold: 0.10
debt_equity_low_threshold: 0.5, growth_strong_threshold: 0.15

# Learning Parameters
learning_rate: 5e-04            # Policy learning rate
lr_initial: 3e-5               # Initial LR (schedule)
lr_final: 1.5e-5               # Final LR (schedule)
use_lr_schedule: True
lr_decay_type: "linear"
ent_coef: 0.05                  # Entropy coefficient
total_timesteps: 100000         # Default; scaling_phase below

# Adaptive Hyperparameters (bidirectional: good and poor performance)
adaptive_lr_sharpe_threshold: 2.0      # Maintain higher LR if Sharpe > this
adaptive_lr_return_threshold: 0.03     # Maintain higher LR if Return > 3%
adaptive_lr_boost_factor: 1.5
adaptive_lr_poor_sharpe_threshold: 1.5  # Boost LR when Sharpe < this (poor performance)
adaptive_lr_poor_return_threshold: 0.03
adaptive_lr_poor_boost_factor: 2.0
adaptive_lr_poor_max: 5e-4      # 2026-02-09: So poor branch increases LR (was 1e-4, was reducing 3e-4→1e-4)
adaptive_ent_min_trades: 50
adaptive_ent_boost_factor: 2.0
adaptive_ent_max: 0.04         # Maximum entropy coefficient (cap)

# Base Rewards (scale 0.7x-1.6x with market via _calculate_market_condition_multiplier)
base_trading_incentive: 0.50    # BUY base
sell_base_incentive: 0.48      # SELL base
exploration_bonus: 0.50
signal_alignment_bonus: 0.15
profitable_trade_bonus: 0.35
learning_bonus: 0.05
first_trade_bonus: 0.40
eval_hold_penalty: -0.04       # Hold penalty in evaluation
position_incentive: 0.02

# SELL-Specific
sell_profit_taking_bonus: 0.50
sell_loss_penalty_cap: -1.0     # Balanced with buy_loss_penalty_cap
buy_loss_penalty_cap: -1.0
sell_bull_market_boost: 0.08

# Dynamic Stop Loss (base; then adjusted by volatility + ATR + regime + signals)
base_stop_loss_pct: 0.10       # 10% base stop loss
trailing_stop_pct: 0.08        # 8% trailing stop for profits
# Formula: base + volatility + ATR + regime + signal adjustments (5%-30% range)

# Signal Gating (env: hardcoded in _should_gate_trade_by_signal_confidence; config defines defaults)
high_confidence_threshold: 0.25  # High confidence (full reward) — env uses 0.25 hardcoded
medium_confidence_threshold: 0.10  # Medium (0.7x) — env uses 0.10 hardcoded
low_confidence_exploration: 0.95  # Config; env uses cooldown-based low-confidence (not fixed % random)

# Scaling (Progressive to 1M)
scaling_phase: "250k"
min_sharpe_to_scale: 0.2       # Min Sharpe to increase timesteps
min_return_to_scale: 0.02      # Min return (2%) to scale up
max_drawdown_to_scale: 0.15
timestep_increase_amount: 100000
timestep_decrease_amount: 50000

# Dynamic Market-Condition Rewards
# Base rewards scale 0.7x-1.6x: volatility, trend, regime (see _calculate_market_condition_multiplier)
```

---

## 🐛 **Troubleshooting**

### **Common Issues:**

#### **1. NameError: 'shares_to_buy' is not defined**
- **Cause**: Variable name mismatch in VWAP reward function
- **Fix**: Use `position_size` instead of `shares_to_buy` in BUY section
- **Status**: ✅ Fixed

#### **2. IndentationError in trading_env.py**
- **Cause**: Incorrect indentation in reward functions
- **Fix**: Corrected all indentation levels
- **Status**: ✅ Fixed

#### **3. UnicodeEncodeError in config.py**
- **Cause**: Emoji characters in print statements
- **Fix**: Replace emoji with plain text
- **Status**: ⚠️ User rejected fix

#### **4. Portfolio Value Mismatches**
- **Cause**: Floating-point precision issues
- **Impact**: Warning messages, not critical
- **Status**: ⚠️ Monitoring

#### **5. NoneType Reward Errors**
- **Cause**: `_execute_trade()` returning `None` in some code paths
- **Fix**: Added proper `else` clause for insufficient funds case
- **Status**: ✅ Fixed

#### **6. Indentation Errors After Formatting**
- **Cause**: Autopep8 formatter created structural problems
- **Fix**: Corrected all indentation levels and control flow
- **Status**: ✅ Fixed

#### **7. UnboundLocalError: 'ticker' in HOLD Action**
- **Cause**: `ticker` variable not defined in HOLD action path
- **Error**: `UnboundLocalError: cannot access local variable 'ticker'` at line 3380
- **Fix**: Changed to check all tickers for strong signals (HOLD affects all tickers)
- **Location**: `ppo/models/trading_env.py` lines 3377-3405
- **Status**: ✅ Fixed

#### **8. Agent Unlearning BC Knowledge**
- **Cause**: PPO training overwrites BC initialization within 10k-20k timesteps
- **Impact**: Agent reverts from trading behavior (BC) to holding behavior
- **Fix**: Implemented BC Regularization callback that preserves BC for 50k timesteps
- **Mechanism**: Monitors policy drift, adjusts learning rate to preserve BC knowledge
- **Location**: `ppo/training/ppo_trainer.py` lines 26-93, 357-362
- **Status**: ✅ Fixed

#### **9. Alpha/Beta always 0; Max consecutive wins/losses = 0 (2026-02-08)**
- **Cause**: Benchmark series was all zeros (no env benchmark_return); consecutive win/loss streaks were never computed.
- **Fix**: Evaluator builds benchmark from SPY or equal-weighted universe step returns; alpha/beta only when benchmark variance > 0; report shows N/A when invalid. Max consecutive wins/losses computed from closing-trade PnL and written to results.
- **Location**: `ppo/evaluation/performance_evaluator.py`, `ppo/evaluation/unified_report_generator.py`
- **Status**: ✅ Fixed (re-run evaluation to refresh reports)

#### **10. Adaptive LR (poor) was reducing LR instead of increasing (2026-02-09)**
- **Cause**: `adaptive_lr_poor_max` was 1e-4, so when current LR was 3e-4, `new_lr = min(1e-4, 6e-4) = 1e-4` (decrease).
- **Fix**: `adaptive_lr_poor_max` set to 5e-4 in `ppo/utils/dynamic_config.py`; in `PPO_fixed.py` poor branch uses `new_lr = max(current_lr, min(poor_lr_max, current_lr * factor))` so LR never drops.
- **Status**: ✅ Fixed

#### **11. Agent not trading enough – deterministic collapse (2026-02-20)**
- **Symptom**: Deterministic eval (paper-trading style) shows **HOLD 93–96%**, **BUY 1–2%**, **4–6 trades**; stochastic eval shows 30+ trades and positive return. PPO is a high-frequency algo but the agent under-trades in deterministic mode.
- **Root cause (primary)**: **RegimeController** (`ppo/utils/regime_controller.py`) reduces **HOLD penalty** in volatile regimes: when `avg_volatility >= 0.02` (2%), `hold_penalty_mult *= 0.75`. So during training, many steps are classified as volatile → HOLD is less penalized → policy learns that HOLD is relatively better → **argmax action becomes HOLD**. At eval, `model.predict(..., deterministic=True)` returns argmax → hence 96% HOLD.
- **Not gating**: Unified report shows "Gating blocked: 0"; with `tech_only_soft_gating=True` and `fundamental_signal_weight < 0.2`, gating allows all trades (reward multiplier only). So the bottleneck is **policy output**, not env blocking.
- **Fixes (apply with approval)**: (1) **RegimeController**: In `regime_controller.py` `compute()`, avoid reducing HOLD penalty below 1.0 in volatile regimes (e.g. use `hold_mult = max(0.75, hold_mult * 0.9)` or remove the 0.75x volatile branch so HOLD is not made more attractive). (2) **Stronger base HOLD penalty**: In `dynamic_config.py`, consider increasing `hold_penalty` / `hold_penalty_no_positions` (e.g. -0.10 → -0.12) so HOLD is costlier during training. (3) **Eval collapse guardrail**: If deterministic BUY% < 5% for N evals, optionally increase `eval_hold_penalty` or apply a one-off exploration nudge (document in REWARD_CHANGELOG).
- **Status**: ⚠️ Identified; fixes pending your approval

---

## 📊 **Performance History**

### **Phase 1 Results:**
- **Return**: 0% (agent not trading)
- **Issue**: Penalties preventing trading
- **Status**: ✅ Resolved

### **Phase 2 Results:**
- **Return**: 14.64% (breakthrough!)
- **Trades**: 1,310 total
- **Win Rate**: 76.9%
- **Status**: ✅ Major improvement

### **Phase 3 Results:**
- **Return**: 1.40% (dropped due to realistic costs)
- **Issue**: Agent learned from artificial high costs
- **Status**: ✅ Optimized with 20x stronger rewards

### **Phase 4 Results:**
- **Return**: -0.53% (temporary drop during integration)
- **Issue**: Agent learning new technical/fundamental signals
- **Status**: ✅ Technical & fundamental analysis integrated
- **Expected**: Significant improvement in next training run

### **Phase 5 Results:**
- **Trade Frequency**: 102 trades (vs previous 8) - 12.75x improvement
- **Return**: 68.49% (monitoring - early results)
- **Win Rate**: 20.8% (22/106 trades)
- **Status**: ✅ Reward structure fixed, BC regularization active
- **Key Fix**: Signals now drive 85% of reward (was 35%)
- **Key Fix**: BC knowledge preserved for 50k timesteps
- **Expected**: Agent will improve trading decisions over time with proper reward structure

---

## 🎯 **Success Metrics**

### **Trading Performance:**
- ✅ **Agent is Trading**: 1,310+ trades executed
- ✅ **Profitable**: 14.64% returns achieved
- ✅ **High Win Rate**: 76.9% success rate
- ✅ **Active Trading**: 131 trades per episode
- ✅ **Realistic Costs**: Alpaca-compliant fees

### **Technical Achievements:**
- ✅ **No Crashes**: All syntax errors fixed
- ✅ **Optimized Rewards**: 20x stronger profit signals
- ✅ **VWAP Intelligence**: BC strategy matching
- ✅ **Realistic Simulation**: Real Alpaca costs
- ✅ **Technical Analysis**: RSI, MACD, Bollinger Bands, Volume
- ✅ **Fundamental Analysis**: PE, ROE, Debt-to-Equity, Growth
- ✅ **Market Regime Detection**: Trending vs Mean-reversion
- ✅ **Reward Shaping**: Multi-factor reward system
- ✅ **Scalable Architecture**: Ready for Phase 5

---

## 🚀 **Future Roadmap**

### **Phase 6: Performance Optimization & Advanced Stop Loss** ✅ (COMPLETE)
- ✅ 10-20x performance optimization (signal caching, vectorization, logging)
- ✅ Advanced dynamic stop loss (volatility + ATR + regime + signals)
- ✅ Signal-confidence gating system (quality over quantity)
- ✅ BUY/SELL reward rebalancing (equal incentives)
- ✅ Signal-aware stop loss rewards (learn from warnings)

### **Phase 7: Validation & Scaling to 1M** (Current)
- Scale 250k → 400k → 600k → 800k → 1M (progressive with metrics monitoring)
- Target >30% returns, Sharpe >1.0, Drawdown <10%
- Validate advanced stop loss effectiveness
- Monitor signal gating impact on trade quality

### **Phase 8: Paper Trading Ready (TG5 Integration)**
- Real-time data feeds integration
- Live market simulation with Alpaca Paper API
- Order execution and position tracking
- Performance monitoring dashboard

### **Phase 9: Production Deployment**
- Real money trading with risk management
- Portfolio allocation strategies
- Multi-asset expansion beyond 11 tickers (if needed)
- Continuous learning and model updates

---

## 📞 **Support & Contact**

For questions, issues, or contributions:
- Review the troubleshooting section above
- Check console output for error messages
- Monitor performance reports in `ppo/daily_reports/`
- Ensure all dependencies are installed

---

## 🏆 **Achievement Summary**

**🎯 PHASE 7 IN PROGRESS: Adaptive Hyperparameter Optimization** ✅ (LATEST)

### **🔄 Adaptive Hyperparameter Systems (PHASE 7 - LATEST):**
✅ **🧠 TCN Memory Architecture**: Upgraded to TCN (Temporal Convolutional Network) policy with dilations [1, 2, 4, 8] for temporal sequence learning (more powerful than LSTM, faster on CPU)  
✅ **⚡ BC Regularization Optimization**: 99.9% reduction in computation (check every 1000 steps), 90% lighter penalty (5% vs 50% LR reduction), weight 0.03 (was 0.3)  
✅ **📊 Evaluation Frequency Optimization**: Increased from 50k to 100k steps (50% reduction in overhead)  
✅ **📝 Progress Logging Optimization**: Increased from 10k to 25k steps (60% reduction in I/O)  
✅ **Adaptive Learning Rate**: Maintains higher LR (1.5x boost) when Sharpe > 2.0 AND Return > 3%  
✅ **Dynamic LR Schedule**: Real-time config reading, adjustments take effect immediately  
✅ **Adaptive Entropy**: Increases entropy (cap 0.04) when trade count < 50 to encourage exploration  
✅ **Self-Correcting**: Automatically resets when performance/trade conditions normalize  
✅ **Problem Solved**: Prevents over-conservatism at 300K+ timesteps (was causing 78% return drop)  
✅ **Configurable Thresholds**: All thresholds adjustable in `dynamic_config.py`  
✅ **Dynamic Market-Condition Rewards**: Base rewards scale 0.7x-1.6x based on volatility, trend, and regime  
✅ **Reward Config Sync Fix**: Fixed critical bug - base rewards now correctly sync from config (0.40 vs 0.08)  
✅ **Reward Balance**: Base rewards BUY 0.50, SELL 0.48, exploration 0.50; eval hold penalty -0.04
✅ **Signal Gating**: High >0.25, medium >0.10 (hardcoded in env); low-confidence cooldown-based
✅ **SELL Reward Enhancement**: sell_base_incentive 0.48, sell_profit_taking_bonus 0.50, sell_bull_market_boost 0.08; loss penalties -1.0 (both)
✅ **Gentle Market Multipliers**: Bull/Bear adjustments gentle (5-10% tilt)
✅ **Bull Market Profit-Taking**: SELL gets +0.08 bonus in bull markets
✅ **SHAP Explainability System**: Automatic SHAP report generation after each evaluation - explains agent decisions with feature importance analysis  

**Key Insight (Phase 7)**: Fixed the 300K timestep regression (LR decay → over-conservatism → trade drop). Adaptive LR is bidirectional (boost when good or poor); entropy cap 0.04 when trades low. Rewards sync from dynamic_config (BUY 0.50, SELL 0.48, exploration 0.50) and scale 0.7x-1.6x with market. RL rewards are technical-only (fundamental_signal_weight=0). Stop loss base 10%, trailing 8%. TCN policy for temporal memory (15+ steps); BC regularization and evaluation/logging optimized.

### **⚡ Performance Breakthroughs (Phase 6):**
✅ **10-20x Speed Improvement**: 5 sec/step → 0.25-0.5 sec/step (347 hours → 17-35 hours for 250k)  
✅ **Signal Caching Optimization**: 71 calls/step → 10 calls/step (7x fewer calculations)  
✅ **Vectorized Portfolio Calc**: NumPy-based with step caching (5-10x faster)  
✅ **Config Value Caching**: One-time load (50% faster technical signals)  
✅ **Reduced Logging**: Every 20th trade (was 5th) - 4x faster I/O  

### **🎯 Advanced Stop Loss System (Phase 6):**
✅ **Volatility-Adjusted**: 20-day volatility widens stop in volatile markets (0-2%)  
✅ **ATR Integration**: Adapts to instrument-specific volatility (0-3%)  
✅ **Market Regime Aware**: Bull (-1%), Bear (+2%), Volatile (+4%) adjustments  
✅ **Signal-Aware**: Tighter stop when signals warn (-2%), wider for strong positives (+1%)  
✅ **Dynamic Penalties**: Small penalty (-0.1) if warned, full penalty (-0.5) if unexpected  
✅ **Formula**: Base 10% + volatility + ATR + regime + signals = 5-30% dynamic range  

### **🔄 Signal Quality & Balance (Phase 6 & 7):**
✅ **Signal-Confidence Gating**: High (>0.25) = 1.0x, Medium (>0.10) = 0.7x (hardcoded in env); low-confidence cooldown-based
✅ **BUY/SELL Rebalance**: base_trading_incentive 0.50, sell_base_incentive 0.48, exploration_bonus 0.50; signal_alignment_bonus 0.15
✅ **Signal Alignment Bonus**: From config; tech + fund when fundamentals enabled (RL currently technical-only)
✅ **Profitable Trade Bonus**: profitable_trade_bonus 0.35 (config); scaled by PnL for both
✅ **SELL-Specific**: sell_profit_taking_bonus 0.50, sell_bull_market_boost 0.08; loss penalties -1.0 (both)

### **Previous Achievements (Phase 5):**
✅ **Reward Structure**: RL rewards technical-only (technical_signal_weight=1.0, fundamental_signal_weight=0.0; fundamental reserved for supervised nets later)
✅ **BC Regularization**: Preserves BC knowledge for 50k timesteps  
✅ **Signal Amplification**: Strong signals get 1.5x-2.0x multiplier  
✅ **Trade Frequency**: 102 trades vs previous 8 (12.75x improvement)  

**Key Insight (Phase 6)**: Training speed is no longer a bottleneck. Advanced stop loss teaches agent to respect signals predicting drops. Signal gating ensures quality over quantity. Balanced rewards eliminate BUY-bias.

**Key Insight (Phase 7)**: Adaptive hyperparameters prevent the over-conservatism problem that caused performance regression at high timesteps. System now self-corrects to maintain optimal trading activity throughout long training runs.

**Next Goal**: Train NEW model with TCN architecture (works with standard PPO, no model incompatibility). Train to 150k with all optimizations (TCN temporal memory, optimized BC regularization, adaptive hyperparameters, SELL reward enhancements), validate improvements, then scale progressively to 1M timesteps for super agent status (>30% return, Sharpe >1.0). Monitor BUY/SELL balance - should see more SELL actions (target: 30-40% SELL vs current 7.5%) as agent learns to take profits. Expected training speed: 2-3x slower than MLP but 2x faster than LSTM on CPU (0.5-1.0 steps/second vs 0.25-0.5) due to TCN parallelization and BC regularization optimizations.

---

*Last Updated: February 2026*
*Status: Phase 7 In Progress - TCN Memory Architecture + Training Optimizations + Adaptive Hyperparameter Optimization + Dynamic Market-Condition Rewards + SELL Reward Enhancement + SHAP Explainability + CSV Data Pipeline + Pre-Computed Calculations Cache + Intraday CSV + Multi-Timeframe Plan*
*Current Phase: Phase 7 - Architecture Upgrade (TCN Temporal Memory) + Training Efficiency (BC Reg Optimization, Evaluation/Logging Frequency) + Data Pipeline Optimization (CSV + Calculation Cache) + Scaling to 1M Timesteps + SHAP Explainability Integration + Multi-Timeframe Data Plan (1m, 5m, 30m, 1h, 1d).*
*Latest Update (Feb 2026): Documentation aligned with dynamic_config.py and config.py: 11 tickers (BTC-USD not on Alpaca), RL rewards technical-only (fundamental_signal_weight=0), base stop 10% / trailing 8%, current reward and scaling values (base_trading_incentive 0.50, sell_base_incentive 0.48, exploration_bonus 0.10, adaptive_ent_max 0.04, scaling_phase "250k", min_sharpe_to_scale 0.2, min_return_to_scale 0.02). Signal gating documented as hardcoded in env (high 0.25, medium 0.10; low-confidence cooldown-based). Key Parameters block and Phase 7 narrative updated to match code.*
*2026-02-08: Added evaluator/report fixes (alpha/beta from real benchmark, max consecutive wins/losses), Sharpe tracking note, and Feature set & noise subsection (intraday vs daily features, noise sources, optional reductions).*
*2026-02-09: Quant improvements – adaptive LR (poor) fix (adaptive_lr_poor_max 5e-4, poor branch never reduces LR); SELL signal_alignment_bonus tied to profitable SELL only; activity bonuses reduced (trading_frequency 0.03, progressive 0.002/0.08, exploration 0.10); hold_winner_bonus 0.05; per-step config cache in env (_cfg()); step-log default 400; test file ppo/tests/test_quant_improvements_2026_02_09.py; docs QUANT_RESEARCH_AGENT_PERFORMANCE_2026-02-09.md, REPORT_ERRORS_2026-02-09.md, ppo/docs/PERFORMANCE_REFINEMENTS_2026-02-09.md.*