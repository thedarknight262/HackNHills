import pandas as pd
import numpy as np

DEFAULT_COST_BPS = 18.0


def run_backtest(
    signals_csv="mcx_gold_signals.csv",
    pair="Petal_vs_Ten",
    min_edge_pct=0.35,
    cost_bps=DEFAULT_COST_BPS,
    execution_mode="T+0_Settlement",  # "T+0_Settlement", "T+1_Realistic_Slippage", or "T+1_Smart_Limit_Order"
    short_only=False,
    output_csv="mcx_backtest_results.csv"
):
    df = pd.read_csv(signals_csv)
    base_cost_pct = cost_bps / 10000.0

    # CRITICAL PDF COMPLIANCE:
    # For Petal_vs_GOLDM, the signal is evaluated on the curve-adjusted Gap_Petal_vs_GOLDM_Pct,
    # but trade execution and PnL use GOLDM_Actual_Price ("using the prices of the contracts actually held").
    pair_mapping = {
        "Petal_vs_Ten":    ("GOLDPETAL",  "GOLDTEN",            "Gap_Petal_vs_Ten_Pct",    "GOLDPETAL_Vol",  "GOLDTEN_Vol"),
        "Guinea_vs_Ten":   ("GOLDGUINEA", "GOLDTEN",            "Gap_Guinea_vs_Ten_Pct",   "GOLDGUINEA_Vol", "GOLDTEN_Vol"),
        "Petal_vs_Guinea": ("GOLDPETAL",  "GOLDGUINEA",         "Gap_Petal_vs_Guinea_Pct", "GOLDPETAL_Vol",  "GOLDGUINEA_Vol"),
        "Petal_vs_GOLDM":  ("GOLDPETAL",  "GOLDM_Actual_Price", "Gap_Petal_vs_GOLDM_Pct",  "GOLDPETAL_Vol",  "GOLDM_Vol"),
    }

    if pair not in pair_mapping:
        raise ValueError(f"Unknown pair: {pair}")

    p1_col, p2_col, gap_col, v1_col, v2_col = pair_mapping[pair]
    z_col = f"{pair}_ZScore"
    mean_col = f"{pair}_RollMean"
    liquid_col = f"{pair}_Liquid"

    trades = []
    expiries = df["Expiry_Date"].dropna().unique()

    for exp in expiries:
        sub = df[df["Expiry_Date"] == exp].sort_values("Date_Parsed").reset_index(drop=True)
        in_pos = False
        pos_type = None
        entry_idx = None
        entry_date = None
        p1_entry = 0.0
        p2_entry = 0.0
        entry_gap = 0.0
        entry_extra_slip_pct = 0.0

        i = 0
        while i < len(sub):
            row = sub.iloc[i]
            d = row["Date_Parsed"]
            dte = row["Days_To_Expiry"]
            p1 = row[p1_col]
            p2 = row[p2_col]
            z = row[z_col]
            mean = row[mean_col]
            gap = row[gap_col]
            liquid = row[liquid_col]

            if pd.isna(p1) or pd.isna(p2) or pd.isna(z):
                i += 1
                continue

            edge = abs(gap - mean) if not pd.isna(mean) else 0.0

            # 1. Entry Logic
            if not in_pos:
                cond_short = (z >= 2.0 and edge >= min_edge_pct)
                cond_long = (not short_only) and (z <= -2.0 and edge >= min_edge_pct)

                if liquid and dte >= 5 and (cond_short or cond_long):
                    cand_pos = "SHORT_SPREAD" if cond_short else "LONG_SPREAD"

                    if execution_mode == "T+1_Realistic_Slippage":
                        # Naive T+1 Market Order: fills next day regardless of overnight decay or thin far-month depth
                        if i + 1 < len(sub) and not pd.isna(sub.iloc[i + 1][p1_col]) and not pd.isna(sub.iloc[i + 1][p2_col]):
                            fill_row = sub.iloc[i + 1]
                            in_pos = True
                            pos_type = cand_pos
                            entry_idx = i + 1
                            entry_date = fill_row["Date_Parsed"]
                            p1_entry = fill_row[p1_col]
                            p2_entry = fill_row[p2_col]
                            entry_gap = fill_row[gap_col]
                            min_vol = max(min(fill_row[v1_col], fill_row[v2_col]), 50.0)
                            extra_bps = 10.0 * (500.0 / min_vol) if min_vol < 500 else 0.0
                            entry_extra_slip_pct = extra_bps / 10000.0
                            i += 2
                            continue

                    elif execution_mode == "T+1_Smart_Limit_Order":
                        # Institutional T+1 Limit Order: requires Peak Liquidity (7 <= DTE <= 75, Vol >= 400)
                        # AND verifies that at least 65% of the required dislocation persists on Day T+1
                        if 7 <= dte <= 75 and min(row[v1_col], row[v2_col]) >= 400:
                            if i + 1 < len(sub) and not pd.isna(sub.iloc[i + 1][p1_col]) and not pd.isna(sub.iloc[i + 1][p2_col]):
                                fill_row = sub.iloc[i + 1]
                                f_gap = fill_row[gap_col]
                                f_edge = (f_gap - mean) if cand_pos == "SHORT_SPREAD" else (mean - f_gap)
                                min_vol = min(fill_row[v1_col], fill_row[v2_col])
                                if f_edge >= (min_edge_pct * 0.65) and min_vol >= 400:
                                    in_pos = True
                                    pos_type = cand_pos
                                    entry_idx = i + 1
                                    entry_date = fill_row["Date_Parsed"]
                                    p1_entry = fill_row[p1_col]
                                    p2_entry = fill_row[p2_col]
                                    entry_gap = f_gap
                                    entry_extra_slip_pct = 0.0
                                    i += 2
                                    continue

                    else:
                        # T+0_Settlement (Idealized same-day settlement fill)
                        in_pos = True
                        pos_type = cand_pos
                        entry_idx, entry_date, p1_entry, p2_entry, entry_gap = i, d, p1, p2, gap
                        entry_extra_slip_pct = 0.0

            # 2. Exit Logic
            else:
                holding_days = i - entry_idx
                exit_reason = None

                if pos_type == "SHORT_SPREAD" and (z <= 0.2 or gap <= mean):
                    exit_reason = "MEAN_REVERSION"
                elif pos_type == "LONG_SPREAD" and (z >= -0.2 or gap >= mean):
                    exit_reason = "MEAN_REVERSION"
                elif pos_type == "SHORT_SPREAD" and (z >= 3.8 or (gap - entry_gap) > 1.2):
                    exit_reason = "STOP_LOSS"
                elif pos_type == "LONG_SPREAD" and (z <= -3.8 or (entry_gap - gap) > 1.2):
                    exit_reason = "STOP_LOSS"
                elif dte <= 3:
                    exit_reason = "EXPIRY_ROLL"
                elif holding_days >= 20:
                    exit_reason = "MAX_HOLD"

                if exit_reason:
                    if execution_mode in ("T+1_Realistic_Slippage", "T+1_Smart_Limit_Order") and i + 1 < len(sub) and not pd.isna(sub.iloc[i + 1][p1_col]) and not pd.isna(sub.iloc[i + 1][p2_col]):
                        exit_row = sub.iloc[i + 1]
                        p1_exit, p2_exit, exit_d = exit_row[p1_col], exit_row[p2_col], exit_row["Date_Parsed"]
                        min_vol = max(min(exit_row[v1_col], exit_row[v2_col]), 50.0)
                        exit_extra_slip_pct = (10.0 * (500.0 / min_vol) / 10000.0) if (execution_mode == "T+1_Realistic_Slippage" and min_vol < 500) else 0.0
                    else:
                        p1_exit, p2_exit, exit_d = p1, p2, d
                        exit_extra_slip_pct = 0.0

                    if pos_type == "SHORT_SPREAD":
                        ret1 = -(p1_exit / p1_entry - 1.0)
                        ret2 = (p2_exit / p2_entry - 1.0)
                    else:
                        ret1 = (p1_exit / p1_entry - 1.0)
                        ret2 = -(p2_exit / p2_entry - 1.0)

                    spread_gross = (ret1 + ret2) / 2.0
                    total_cost_pct = base_cost_pct + entry_extra_slip_pct + exit_extra_slip_pct
                    spread_net = spread_gross - total_cost_pct
                    gold_bench = (p2_exit / p2_entry - 1.0)

                    trades.append({
                        "Pair": pair,
                        "Expiry": exp,
                        "Entry_Date": entry_date,
                        "Exit_Date": exit_d,
                        "Position": pos_type,
                        "Holding_Days": holding_days,
                        "Exit_Reason": exit_reason,
                        "Gross_Return_Pct": round(spread_gross * 100.0, 3),
                        "Net_Return_Pct": round(spread_net * 100.0, 3),
                        "Cost_Deducted_Pct": round(total_cost_pct * 100.0, 3),
                        "Gold_Benchmark_Return_Pct": round(gold_bench * 100.0, 3),
                        "Is_Out_Of_Sample": pd.to_datetime(exit_d) >= pd.to_datetime("2025-10-21")
                    })
                    in_pos = False
            i += 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["Exit_Date"] = pd.to_datetime(trades_df["Exit_Date"])
        trades_df["Entry_Date"] = pd.to_datetime(trades_df["Entry_Date"])
        trades_df["Expiry"] = pd.to_datetime(trades_df["Expiry"])

        trades_df = trades_df.sort_values("Exit_Date").reset_index(drop=True)
        trades_df["Cumulative_Gross_Pct"] = trades_df["Gross_Return_Pct"].cumsum().round(2)
        trades_df["Cumulative_Net_Pct"] = trades_df["Net_Return_Pct"].cumsum().round(2)

        trades_df["Entry_Date_Str"] = trades_df["Entry_Date"].dt.strftime("%d %b %Y")
        trades_df["Exit_Date_Str"] = trades_df["Exit_Date"].dt.strftime("%d %b %Y")
        trades_df["Expiry_Str"] = trades_df["Expiry"].dt.strftime("%d %b %Y")

        trades_df.to_csv(output_csv, index=False)
    return trades_df


if __name__ == "__main__":
    run_backtest()