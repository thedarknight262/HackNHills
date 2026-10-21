import pandas as pd
import numpy as np

LOOKBACK_WINDOW = 15
MIN_PERIODS = 10
Z_THRESHOLD = 2.0
MIN_EDGE_PCT = 0.15
MIN_DTE = 5
MIN_VOLUME_LOTS = 100


def build_analytics_and_signals(
    input_csv="mcx_gold_2years.csv",
    output_csv="mcx_gold_signals.csv"
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df["Date_Parsed"] = pd.to_datetime(df["Date_Parsed"])
    df["Expiry_Date"] = pd.to_datetime(df["Expiry_Date"])
    df = df.sort_values(["Date_Parsed", "Expiry_Date", "Symbol"]).reset_index(drop=True)

    eom_syms = ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]
    eom_df = df[df["Symbol"].isin(eom_syms)].copy()

    idx_cols = ["Date_Parsed", "Expiry_Date", "Days_To_Expiry"]
    piv_price = eom_df.pivot(index=idx_cols, columns="Symbol", values="Normalized_Price_10g_999")
    piv_vol = eom_df.pivot(index=idx_cols, columns="Symbol", values="Volume")
    piv_oi = eom_df.pivot(index=idx_cols, columns="Symbol", values="OpenInterest")

    pairs_df = piv_price.reset_index()
    for sym in eom_syms:
        if sym in piv_vol.columns:
            pairs_df[f"{sym}_Vol"] = piv_vol[sym].values
            pairs_df[f"{sym}_OI"] = piv_oi[sym].values

    # Phase 3B: Both Synthetic Curve-Interpolated GOLDM (for clean signal generation)
    # AND Actual Held GOLDM Contract (for realistic trade execution & PnL reporting)
    goldm_df = df[df["Symbol"] == "GOLDM"].sort_values(["Date_Parsed", "Days_To_Expiry"])
    goldm_by_date = {d: g for d, g in goldm_df.groupby("Date_Parsed")}

    synth_goldm_prices = []
    actual_goldm_prices = []
    actual_goldm_expiry = []
    actual_goldm_dte = []
    actual_goldm_vol = []
    implied_carry_pa = []

    for _, row in pairs_df.iterrows():
        d = row["Date_Parsed"]
        target_dte = row["Days_To_Expiry"]
        eom_exp = row["Expiry_Date"]

        if d in goldm_by_date:
            g = goldm_by_date[d]
            g_pos = g[g["Days_To_Expiry"] > 0]

            # 1. Match Actual Held GOLDM Contract (nearest expiry >= EOM expiry, typically 3rd-5th of next month)
            g_after = g_pos[g_pos["Expiry_Date"] >= eom_exp]
            best_actual = g_after.iloc[0] if not g_after.empty else (g_pos.iloc[-1] if not g_pos.empty else None)

            if best_actual is not None:
                actual_goldm_prices.append(round(float(best_actual["Normalized_Price_10g_999"]), 2))
                actual_goldm_expiry.append(best_actual["Expiry_Date"])
                actual_goldm_dte.append(int(best_actual["Days_To_Expiry"]))
                actual_goldm_vol.append(int(best_actual["Volume"]))
            else:
                actual_goldm_prices.append(np.nan)
                actual_goldm_expiry.append(pd.NaT)
                actual_goldm_dte.append(np.nan)
                actual_goldm_vol.append(0)

            # 2. Curve-Interpolated Synthetic GOLDM at exact EOM Days_To_Expiry
            if len(g_pos) >= 2:
                dtes = g_pos["Days_To_Expiry"].values
                prices = g_pos["Normalized_Price_10g_999"].values
                interp_p = np.interp(target_dte, dtes, prices)
                carry = ((prices[1] / prices[0]) - 1.0) * (365.0 / (dtes[1] - dtes[0])) * 100.0
                synth_goldm_prices.append(round(float(interp_p), 2))
                implied_carry_pa.append(round(float(carry), 2))
            elif len(g_pos) == 1:
                synth_goldm_prices.append(round(float(g_pos["Normalized_Price_10g_999"].iloc[0]), 2))
                implied_carry_pa.append(np.nan)
            else:
                synth_goldm_prices.append(np.nan)
                implied_carry_pa.append(np.nan)
        else:
            synth_goldm_prices.append(np.nan)
            actual_goldm_prices.append(np.nan)
            actual_goldm_expiry.append(pd.NaT)
            actual_goldm_dte.append(np.nan)
            actual_goldm_vol.append(0)
            implied_carry_pa.append(np.nan)

    pairs_df["GOLDM_Synthetic_EOM"] = synth_goldm_prices
    pairs_df["GOLDM_Actual_Price"] = actual_goldm_prices
    pairs_df["GOLDM_Actual_Expiry"] = actual_goldm_expiry
    pairs_df["GOLDM_Actual_DTE"] = actual_goldm_dte
    pairs_df["GOLDM_Vol"] = actual_goldm_vol
    pairs_df["Implied_Annual_Carry_Pct"] = implied_carry_pa

    # Relative Price Gaps (%)
    pairs_df["Gap_Petal_vs_Ten_Pct"] = ((pairs_df["GOLDPETAL"] / pairs_df["GOLDTEN"]) - 1.0) * 100.0
    pairs_df["Gap_Guinea_vs_Ten_Pct"] = ((pairs_df["GOLDGUINEA"] / pairs_df["GOLDTEN"]) - 1.0) * 100.0
    pairs_df["Gap_Petal_vs_Guinea_Pct"] = ((pairs_df["GOLDPETAL"] / pairs_df["GOLDGUINEA"]) - 1.0) * 100.0

    # Decomposition: Raw Gap vs Actual Held GOLDM = Genuine Curve-Adjusted Basis + Mechanical Roll-Down Carry Wedge
    pairs_df["Raw_Gap_Petal_vs_Actual_GOLDM_Pct"] = ((pairs_df["GOLDPETAL"] / pairs_df["GOLDM_Actual_Price"]) - 1.0) * 100.0
    pairs_df["Gap_Petal_vs_GOLDM_Pct"] = ((pairs_df["GOLDPETAL"] / pairs_df["GOLDM_Synthetic_EOM"]) - 1.0) * 100.0
    pairs_df["Mechanical_Carry_Wedge_Pct"] = (
        pairs_df["Raw_Gap_Petal_vs_Actual_GOLDM_Pct"] - pairs_df["Gap_Petal_vs_GOLDM_Pct"]
    ).round(4)

    pairs_df["Gap_Guinea_vs_GOLDM_Pct"] = ((pairs_df["GOLDGUINEA"] / pairs_df["GOLDM_Synthetic_EOM"]) - 1.0) * 100.0
    pairs_df["Gap_Ten_vs_GOLDM_Pct"] = ((pairs_df["GOLDTEN"] / pairs_df["GOLDM_Synthetic_EOM"]) - 1.0) * 100.0

    # Walk-Forward Z-Score Engine (strictly grouped by Expiry_Date and shifted by t-1)
    pairs_df = pairs_df.sort_values(["Expiry_Date", "Date_Parsed"]).reset_index(drop=True)

    spread_definitions = {
        "Petal_vs_Ten":    ("Gap_Petal_vs_Ten_Pct",    "GOLDPETAL_Vol",  "GOLDTEN_Vol"),
        "Guinea_vs_Ten":   ("Gap_Guinea_vs_Ten_Pct",   "GOLDGUINEA_Vol", "GOLDTEN_Vol"),
        "Petal_vs_Guinea": ("Gap_Petal_vs_Guinea_Pct", "GOLDPETAL_Vol",  "GOLDGUINEA_Vol"),
        "Petal_vs_GOLDM":  ("Gap_Petal_vs_GOLDM_Pct",  "GOLDPETAL_Vol",  "GOLDM_Vol"),
    }

    for pair_name, (gap_col, vol_col_a, vol_col_b) in spread_definitions.items():
        grouped_gap = pairs_df.groupby("Expiry_Date")[gap_col]
        roll_mean = grouped_gap.transform(
            lambda s: s.shift(1).rolling(LOOKBACK_WINDOW, min_periods=MIN_PERIODS).mean()
        )
        roll_std = grouped_gap.transform(
            lambda s: s.shift(1).rolling(LOOKBACK_WINDOW, min_periods=MIN_PERIODS).std()
        )

        z_score = (pairs_df[gap_col] - roll_mean) / roll_std.replace(0, np.nan)
        edge_pct = (pairs_df[gap_col] - roll_mean).abs()

        is_liquid = (
            (pairs_df["Days_To_Expiry"] >= MIN_DTE) &
            (pairs_df[vol_col_a].fillna(0) >= MIN_VOLUME_LOTS) &
            (pairs_df[vol_col_b].fillna(0) >= MIN_VOLUME_LOTS)
        )

        signal = np.where(
            is_liquid & (z_score >= Z_THRESHOLD) & (edge_pct >= MIN_EDGE_PCT),
            "SHORT_SPREAD",
            np.where(
                is_liquid & (z_score <= -Z_THRESHOLD) & (edge_pct >= MIN_EDGE_PCT),
                "LONG_SPREAD",
                "QUIET"
            )
        )

        pairs_df[f"{pair_name}_RollMean"] = roll_mean.round(4)
        pairs_df[f"{pair_name}_RollStd"] = roll_std.round(4)
        pairs_df[f"{pair_name}_ZScore"] = z_score.round(2)
        pairs_df[f"{pair_name}_Liquid"] = is_liquid
        pairs_df[f"{pair_name}_Signal"] = signal

    pairs_df = pairs_df.sort_values(["Date_Parsed", "Expiry_Date"]).reset_index(drop=True)
    pairs_df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved {len(pairs_df)} rows with Actual Held GOLDM & Roll-Down Decomposition to {output_csv}")
    return pairs_df


if __name__ == "__main__":
    build_analytics_and_signals()