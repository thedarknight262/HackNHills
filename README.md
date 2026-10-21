# 🪙 Commodity Derivatives Intelligence — MCX Gold Futures
**Hack in Hills ’26 | Problem Statement 03: Commodity Derivatives Intelligence**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Data](https://img.shields.io/badge/MCX%20Bhavcopy-9%2C616%20Contract--Days-success.svg)](https://www.mcxindia.com/market-data/bhavcopy)
[![Validation](https://img.shields.io/badge/Look--Ahead%20Bias-0.000000%25-brightgreen.svg)](#%EF%B8%8F-automated-5-point-verification-suite)

An institutional-grade relative-value analytics, term-structure interpolation, and walk-forward validation framework for Indian gold futures on the **Multi Commodity Exchange of India (MCX)**: **`GOLDM`**, **`GOLDTEN`**, **`GOLDGUINEA`**, and **`GOLDPETAL`**.

---

## 📌 1. Executive Summary & Problem Statement

MCX lists four distinct futures contracts tied to the same underlying physical metal (Gold). However, their raw exchange prices cannot be compared directly due to three structural differences:
1. **Heterogeneous Quotation Units & Lot Sizes:** Quoted per 10g (`GOLDM`, `GOLDTEN`), 8g (`GOLDGUINEA`), or 1g (`GOLDPETAL`), with trading units ranging from 1g to 100g.
2. **Purity Basis Discrepancies:** `GOLDM` is specified at **995 purity**, whereas `GOLDTEN`, `GOLDGUINEA`, and `GOLDPETAL` are specified at **999 purity**.
3. **Asynchronous Maturity Windows:** `GOLDTEN`, `GOLDGUINEA`, and `GOLDPETAL` expire at the **end of the month (27th–31st)**, while `GOLDM` expires at the **start of the month (3rd–5th)**. Comparing raw normalized prices across mismatched expiries conflates normal futures term-structure carry with genuine mispricing.

This project builds a complete pipeline from raw MCX Bhavcopy ingestion to a 7-tab interactive Streamlit intelligence suite. It normalizes all contracts to a common basis, strips out mechanical maturity carry via linear curve interpolation, generates strictly causal (`t-1` shifted) statistical alerts, and tests whether observed spreads survive realistic Indian exchange transaction costs, execution delays, and thin-volume slippage.

---

## ⚖️ 2. Contract Specifications & Price Normalization

| Contract Symbol | Trading Unit (1 Lot) | Price Quote Basis | Listed Purity | Expiry Window | History in Dataset |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`GOLDM`** | 100 grams | ₹ per 10 grams | **995** | **3rd–5th** (Start-of-Month) | Full 2 Years (2,370 rows) |
| **`GOLDTEN`** | 10 grams | ₹ per 10 grams | **999** | **27th–31st** (End-of-Month) | From Mar 31, 2025 listing (2,030 rows) |
| **`GOLDGUINEA`** | 8 grams | ₹ per 8 grams | **999** | **27th–31st** (End-of-Month) | Full 2 Years (2,608 rows) |
| **`GOLDPETAL`** | 1 gram | ₹ per 1 gram | **999** | **27th–31st** (End-of-Month) | Full 2 Years (2,608 rows) |

### Common Normalization Formula
Every contract settlement price is converted to the equivalent value of **10 grams of 999 purity gold**:

$$\text{Normalized Price}_{10\text{g}, 999} = \text{Settlement Price} \times \left(\frac{10}{\text{Quoted Grams}}\right) \times \left(\frac{999}{\text{Purity}}\right)$$

### Physical Gram-Neutral Hedge Ratios
Because contract lot sizes differ in physical weight, spread execution requires exact gram-neutral lot ratios:
* **`Petal_vs_Ten` (10g unit):** `10 Lots GOLDPETAL` (10g) vs. `1 Lot GOLDTEN` (10g)
* **`Guinea_vs_Ten` (40g unit):** `5 Lots GOLDGUINEA` (40g) vs. `4 Lots GOLDTEN` (40g)
* **`Petal_vs_Guinea` (8g unit):** `8 Lots GOLDPETAL` (8g) vs. `1 Lot GOLDGUINEA` (8g)
* **`Petal_vs_GOLDM` (100g unit):** `100 Lots GOLDPETAL` (100g) vs. `1 Lot GOLDM` (100g)

---

## 🔬 3. Key Empirical Findings (Oct 2024 – Oct 2026 | 514 Trading Days)

### Finding 1: Persistent Retail Micro-Contract Markup
Across **2,030 matched same-expiry contract-days**, retail micro-contracts systematically trade at a structural premium over institutional benchmarks:
* **`GOLDPETAL` vs `GOLDTEN`:** Mean gap of **+0.95%** (Median **+0.76%**), driven by retail accumulation and indivisible lot convenience.
* **`GOLDGUINEA` vs `GOLDTEN`:** Mean gap of **+0.87%** (Median **+0.67%**).
* **`GOLDTEN` vs Curve-Interpolated `GOLDM`:** Median gap of just **+0.02%**, confirming that institutional 10g and 100g contracts price along the same arbitrage-free institutional curve.

### Finding 2: Separating Mechanical Roll-Down from Genuine Basis
Because `GOLDM` expires on the 3rd–5th of the following month (~5–6 days after end-of-month contracts), our engine computes two distinct series for `GOLDM`:
1. **`GOLDM_Synthetic_EOM`:** Linearly interpolated along the `GOLDM` futures curve onto the exact `Days_To_Expiry` of the end-of-month contracts. Used exclusively to measure **Genuine Curve-Adjusted Basis**.
2. **`GOLDM_Actual_Price`:** The price of the nearest actually tradable `GOLDM` contract. Used exclusively for **Trade Execution & PnL Attribution**, adhering to the rule that synthetic interpolated prices cannot be held in live markets.
3. **`Mechanical_Carry_Wedge_Pct`:** Explicitly isolates the portion of the raw price spread caused purely by maturity mismatch ($\text{Raw Gap} - \text{Curve-Adjusted Gap}$).

### Finding 3: Ornstein-Uhlenbeck Mean-Reversion Econometrics
Fitting a discrete Ornstein-Uhlenbeck AR(1) process ($\Delta s_t = \alpha + \lambda s_{t-1} + \varepsilon_t$) within each individual contract expiry proves that same-expiry spreads mean-revert **2.5x faster** than cross-curve spreads:

| Spread Pair | Contract-Days | AR(1) Persistence ($\rho$) | Reversion Speed ($\lambda$) | **OU Half-Life ($\tau_{1/2}$)** | **Dickey-Fuller $t$-Stat** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Guinea_vs_Ten` (Same Expiry)** | 2,006 | 0.774 | -0.226 | **2.71 Trading Days** | **-17.68** ($p < 0.0001$) |
| **`Petal_vs_Ten` (Same Expiry)** | 2,006 | 0.781 | -0.219 | **2.81 Trading Days** | **-17.15** ($p < 0.0001$) |
| **`Petal_vs_Guinea` (Same Expiry)** | 2,570 | 0.838 | -0.162 | **3.91 Trading Days** | **-16.15** ($p < 0.0001$) |
| **`Petal_vs_GOLDM` (Cross-Curve)** | 2,570 | 0.907 | -0.093 | **7.07 Trading Days** | **-10.21** ($p < 0.0001$) |

### Finding 4: Microstructure Proof of Retail Speculative Churn (`Volume / OI`)
On days when our engine fires a `SHORT_SPREAD` alert ($Z \ge +2.0$), `GOLDPETAL`'s daily **Volume-to-Open-Interest Ratio (`Vol / OI`)** averages **1.46x** vs. **1.13x** on `QUIET` days—a **+29.1% surge in speculative retail turnover** that temporarily pushes `GOLDPETAL` above fair value before reverting.

### Finding 5: The 3-Tier Execution Reality (`T+0` vs `T+1` Naive vs `T+1` Smart Limit)
We model a baseline **18.0 bps (0.18%) round-trip transaction friction** covering Indian statutory costs (CTT at 1.25 bps, MCX exchange turnover fee at 0.42 bps, SEBI charges, Stamp Duty, 18% GST) plus bid-ask spread allowance, and evaluate three execution regimes (`Petal_vs_Ten`, Hurdle $\ge 0.35\%$):

| Execution & Liquidity Regime | Direction Filter | Executed Trades | Gross Return (%) | **Net Return After Costs (%)** | Net Win Rate (%) | Analytical Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. `T+0_Settlement` (Idealized)** | Both Directions | 78 | +16.28% | **+2.24%** | 48.7% | Profitable if official settlement prices are treated as instant fills. |
| **1b. `T+0_Settlement` (Short-Only)** | `SHORT_SPREAD` Only | 50 | +14.62% | **+5.62%** | **62.0%** | Fading retail FOMO spikes (`Z >= +2`) drives virtually all positive net alpha. |
| **2. `T+1_Realistic_Slippage` (Naive Market Order)** | Both Directions | 75 | +13.75% | **-11.88%** | 26.7% | **Edge vanishes** under blind next-day market orders due to overnight decay and thin far-month slippage (up to 200 bps). |
| **3. `T+1_Smart_Limit_Order` (Institutional)** | `SHORT_SPREAD` Only | 16 | +6.09% | **+3.21%** | **50.0%** | **Edge survives `T+1`** when gated to the Peak Liquidity Window (`7 <= DTE <= 75`, `Vol >= 400`) with a next-day limit-order persistence check. |
| **3b. `Same_Expiry_Portfolio` (`T+1` Smart Limit)** | `SHORT_SPREAD` Only | 34 | +10.55% | **+4.43%** | **50.0%** | Combining `Petal_vs_Ten` + `Guinea_vs_Ten` yields **93.4% bootstrap probability** of positive net return. |

---

## 🏗️ 4. Repository Architecture

```text
hack_mcx/
├── mcx_scraper.py             
├── analytics_engine.py        
├── backtest_engine.py         
├── app.py                     
├── mcx_gold_2years.csv        
├── mcx_gold_signals.csv       
├── mcx_backtest_results.csv   
└── README.md                  
```

---

## 🖥️ 5. Interactive Dashboard Walkthrough (`app.py`)

The Streamlit application is organized into **7 specialized tabs**:
1. **📊 Market & Gate Scanner:** Displays the latest exchange Bhavcopy status alongside a **Live 3-Gate Explainable Decision Audit** (`|Z| >= 2.0`, `Edge >= Cost Hurdle`, and `Liquidity/DTE Gate`), showing exactly how the system prevented false-alarm trades on `20 Oct 2026`.
2. **📈 Term Structure & Carry:** Plots cross-contract futures curves by `Days_To_Expiry` and decomposes the `GOLDPETAL` vs. `GOLDM` spread into **Mechanical Maturity Carry Wedge** vs. **Genuine Curve-Adjusted Basis**.
3. **🎯 Z-Score & OU Half-Life:** Visualizes the 15-day `t-1` shifted rolling mean, $\pm 2\sigma$ quiet zone (~93% quiet discipline), and the **Ornstein-Uhlenbeck Half-Life & Dickey-Fuller Stationarity Table**.
4. **🧪 Walk-Forward & Bootstrap:** Compares **Gross vs. Net Cumulative Equity Curves**, **Underwater Drawdown Profiles**, **1,000-Iteration Monte Carlo Bootstrap Distributions**, and the **In-Sample (Oct 2024–Oct 2025) vs. Out-of-Sample (Oct 2025–Oct 2026) Validation Split**.
5. **🗓️ Lifecycle, OI & Sizing:** Maps median **Volume** and **Open Interest** across `Days_To_Expiry` buckets (explaining why entries are blocked in the `< 5 DTE` tender zone) and includes an interactive **Gram-Neutral Lot & SPAN Margin Calculator**.
6. **🔥 Cost Matrix & Attribution:** Runs a **25-backtest Sensitivity Heatmap** (`5 Hurdles × 5 Cost Regimes`), a **Cross-Pair Master Matrix**, and a **PnL Attribution Waterfall Bridge**.
7. **🛡️ Judge Verification Suite:** Executes 5 live mathematical assertions directly on the dataset to verify zero data hygiene or look-ahead violations.

---

## 🛡️ 6. Automated 5-Point Verification Suite

Tab 7 of the dashboard executes five live unit tests on the CSV files:
* **Test 1 — Holiday Fallback & Duplicate Filter:** Confirms `0` duplicate `(Symbol, Date, Expiry)` records and `0` weekend fallback rows across all 9,616 rows.
* **Test 2 — Normalization Formula Audit:** Confirms `₹0.000000` maximum deviation from the exact quantity-and-purity normalization formula across all 9,616 rows.
* **Test 3 — `+500%` Shock Look-Ahead Perturbation Test:** Injects an artificial `+500%` price spike on Day $T$ and verifies that Day $T$'s shifted rolling mean (`t-1`) shifts by exactly **`0.0000000000%`**.
* **Test 4 — Contract-Expiry Isolation Check:** Confirms `0` cross-expiry rolling-window leaks during the initial 10-day warm-up period of every listed contract.
* **Test 5 — Tender-Period & Thin-Liquidity Gate:** Confirms `0` active signals generated when `Days_To_Expiry < 5` or leg volume `< 100` lots.

---

## 🚀 7. Installation & Quickstart

### Prerequisites
Install Python 3.9+ and the required packages:

```bash
pip3 install pandas numpy curl_cffi beautifulsoup4 plotly streamlit
```

### Step 1: (Optional) Scrape Fresh MCX Bhavcopy Data
`mcx_gold_2years.csv` is already included in the repository. To scrape new dates with automatic resume checkpointing:

```bash
python3 mcx_scraper.py
```

### Step 2: Run the Analytics & Backtest Engines
Regenerate the 46-column signal matrix (`mcx_gold_signals.csv`) and verified trade log (`mcx_backtest_results.csv`):

```bash
python3 analytics_engine.py
python3 backtest_engine.py
```

### Step 3: Launch the Interactive Dashboard
Start the local Streamlit web server:

```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ⚠️ Disclaimer
This software was built strictly for academic and analytical evaluation at **Hack in Hills ’26**. All backtests and signals are historical simulations incorporating explicit transaction-cost and liquidity assumptions and do not constitute live financial or trading advice.
