# Stock-AIML: Deep Learning Backtesting & Strategy Optimization Platform

A production-grade, modular backtesting and strategy optimization system for stock strategies. Combines execution-aware simulation, temporal graph neural networks (TGNN) for cross-asset contagion, adversarial stress testing, RL-based optimization, and counterfactual explainability.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   DASHBOARD (Streamlit)                       │
│  Upload · Config · Backtest · Stress · Explain · Export       │
├──────────────────────────────────────────────────────────────┤
│                   FASTAPI REST LAYER                          │
│  POST /backtest  /stress-test  /optimize  /explain            │
├──────────────────────────────────────────────────────────────┤
│  BACKTESTER  │  STRESS ENGINE  │  RL OPTIMIZER  │  XAI       │
│  (execution-aware simulation)   (AMG)         (PPO)          │
├──────────────────────────────────────────────────────────────┤
│           TGNN (Temporal Graph Neural Network)                │
│         Cross-asset contagion embeddings                      │
├──────────────────────────────────────────────────────────────┤
│  EXECUTION SURROGATE  │  FEATURE ENGINEERING  │  DATA LAYER │
│  (parametric fill model)  (technical indicators)             │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Easy 1-Click Local Runner (New)

We've added a one-click script to run both the FastAPI Backend and Streamlit Dashboard simultaneously on your local machine.

- **Windows:** Double-click on `run_local.bat` (or run it via terminal).
- **Mac/Linux:** Run `bash run_local.sh`.

*The script will automatically check Python, create a virtual environment, install requirements, and start both servers for you!*

---

### Manual Setup (Step-by-step)

**1. Install dependencies**

```bash
cd stock-aiml
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Dashboard opens at: **http://localhost:8501**

### 3. Run the FastAPI backend (optional — dashboard runs standalone)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs at: **http://localhost:8000/docs**

---

## Project Structure

```
stock-aiml/
├── backend/
│   ├── main.py                  FastAPI entry point
│   ├── api/
│   │   ├── schemas.py           Pydantic request/response models
│   │   └── routes/
│   │       ├── backtest.py       POST /api/v1/backtest
│   │       ├── stress.py        POST /api/v1/stress-test
│   │       ├── optimize.py       POST /api/v1/optimize
│   │       ├── explain.py        POST /api/v1/explain
│   │       └── data.py           POST /api/v1/data/upload
│   ├── core/
│   │   ├── config.py             config.yaml loader
│   │   └── exceptions.py        custom exception classes
│   └── services/
│       ├── data_loader.py        OHLCV ingestion + walk-forward splits
│       ├── feature_engineering.py returns, RSI, MACD, Bollinger, ATR
│       ├── execution_model.py    fill probability + market impact
│       ├── backtester.py         execution-aware simulation engine
│       ├── metrics.py            CAGR, Sharpe, Sortino, CVaR, etc.
│       ├── stress_generator.py   adversarial scenario injection
│       ├── rl_optimizer.py       PPO agent + grid-search
│       └── counterfactual.py     SCM-based counterfactual analysis
├── ml/
│   ├── tgnn/
│   │   └── model.py              TGNN (GRU/Temporal + Graph Attention)
│   └── adversarial/
│       ├── generator.py          Conditional GAN generator
│       └── discriminator.py      Plausibility critic
├── dashboard/
│   ├── app.py                   Streamlit main (sidebar nav)
│   └── pages/
│       ├── dataset_upload.py     CSV upload + schema preview
│       ├── strategy_config.py    Strategy parameter UI
│       ├── backtest_run.py        Run backtest + live charts
│       ├── results_summary.py     Metrics cards + equity/drawdown
│       ├── stress_test.py         Adversarial scenario runner
│       ├── explanation.py         Counterfactual + attribution
│       └── export.py              JSON/CSV download
├── configs/
│   ├── config.yaml               System-wide configuration
│   └── strategies/
│       ├── momentum.yaml         Momentum strategy spec
│       └── mean_reversion.yaml   Mean reversion strategy spec
├── scripts/
│   ├── run_backtest.py          CLI backtest runner
│   ├── train_tgnn.py            TGNN training script
│   └── generate_report.py        Report generator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Module Guide

### Data Layer
The system ingests OHLCV CSV data. If no volume column exists, it synthesizes a volume proxy from intra-bar range. All data is sorted chronologically and time-aligned across splits.

### Feature Engineering
Computes: log returns (1d, 5d, 10d, 20d), rolling volatility (5d, 10d, 20d), RSI, MACD, Bollinger Bands, ATR, volume ratio, regime flags. **No lookahead** — all features use only past data.

### Walk-Forward Splits
Non-overlapping train/val/test windows with configurable gap to prevent look-ahead leakage. Default: 60/20/20 split across 3 rolling windows.

### Execution Surrogate (Parametric)
```
impact   = 0.5 * (order_size / ADV) * volatility * 1.2
fill_prob = 0.85 * exp(-2 * order_size/ADV)
total_cost (bps) = spread_cost + market_impact_bps
```
Direction matters: BUY executes at worse price, SELL at worse price.

### Backtester
Accepts strategy params (RSI thresholds, lookback windows, stop/take-profit, trailing stop). Simulates fills, tracks positions, computes equity/drawdown curves. Stops on max drawdown breach.

### Stress Testing (AMG)
Generates worst-case scenarios via controlled injection:
- **Liquidity shock**: volume drops 15%, price barely moves
- **Correlated selloff**: smooth exponential price drop over 10 bars
- **Volatility spike**: intra-bar range multiplies 2x
- **Regime shift**: persistent 5% price level shift
- **Fast reversal**: V-shaped price profile (drop + recovery)

### RL Optimizer (PPO)
State = TGNN embedding + portfolio features. Action = parameter adjustments. Reward = Sharpe - λ_cvar*CVaR - λ_turnover*Turnover. Grid-search is the default MVP mode.

### Counterfactual / XAI
Trains a surrogate model on synthetic operational configurations. Intervenes on latency, slippage, order size, and commission. Ranks by expected max-drawdown reduction. Outputs plain-English recommendations.

---

## Supported Strategies

| Strategy | Key Parameters | Signal |
|----------|---------------|--------|
| `momentum` | RSI thresholds, lookback windows | Trend-following (short > long return + RSI) |
| `mean_reversion` | Bollinger bands, z-score thresholds | Fade moves when price diverges from mean |

---

## Key Metrics

| Metric | Formula |
|--------|---------|
| Annualized Return | `exp(mean(log_returns) * 252) - 1` |
| Sharpe Ratio | `mean(excess_returns) / std(returns) * sqrt(252)` |
| Sortino Ratio | `mean(excess_returns) / std(downside_returns) * sqrt(252)` |
| Max Drawdown | `max(peak - equity) / peak` |
| CVaR (95%) | Mean return below 5th percentile |
| Win Rate | `trades_with_pnl > 0 / total_trades` |
| Profit Factor | `abs(wins_sum / losses_sum)` |

---

## Configuration

All parameters live in `configs/config.yaml`. Key sections:

```yaml
backtester:
  initial_capital: 1_000_000.0
  commission_pct: 0.001
  slippage_bps: 5

execution:
  impact_coeff: 0.5
  fill_rate: 0.85
  vol_impact_mult: 1.2

tgnn:
  embedding_dim: 64
  hidden_dim: 128
  num_heads: 4
  window_size: 20

adversarial:
  num_samples: 50
  scenarios:
    - type: correlated_selloff
      magnitude: -0.10
      duration: 10
```

---

## Docker (optional)

```bash
docker build -t stock-aiml .
docker run -p 8501:8501 -p 8000:8000 stock-aiml
```

Or use docker-compose:
```bash
docker-compose up
```

---

## Adding New Strategies

1. Add a signal function to `backend/services/backtester.py` (`STRATEGY_SIGNALS` dict)
2. Create a YAML in `configs/strategies/<name>.yaml`
3. The backtester auto-discovers and runs it

---

## Extending the TGNN

The TGNN model in `ml/tgnn/model.py` accepts `(B, T, N, D)` tensors:
- **B** = batch size
- **T** = temporal window length
- **N** = number of nodes (stocks)
- **D** = node feature dimension

For multi-stock data, pass a dict of DataFrames — the `TGNNDataLoader` handles alignment and correlation-based adjacency construction.

---

## License

MIT






Project structure is clean — empty experiments/ folder deleted, nested duplicate
  directory removed.

  Quick start:
  # Run dashboard
  cd stock-aiml && streamlit run dashboard/app.py

  # Run API server
  cd stock-aiml && uvicorn backend.main:app --reload

  # Run full pipeline
  python stock-aiml/scripts/run_pipeline.py
