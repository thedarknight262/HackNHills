import os
import time
import re
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests

# Contract specifications directly from Hack in Hills '26 problem statement
CONTRACT_SPECS = {
    "GOLDM":      {"quoted_grams": 10, "purity": 995, "trading_unit_g": 100, "expiry_window": "3rd-5th"},
    "GOLDTEN":    {"quoted_grams": 10, "purity": 999, "trading_unit_g": 10,  "expiry_window": "27th-31st"},
    "GOLDGUINEA": {"quoted_grams": 8,  "purity": 999, "trading_unit_g": 8,   "expiry_window": "27th-31st"},
    "GOLDPETAL":  {"quoted_grams": 1,  "purity": 999, "trading_unit_g": 1,   "expiry_window": "27th-31st"},
}

class MCXBhavcopyScraper:
    BASE_URL = "https://www.mcxindia.com/market-data/bhavcopy"
    API_URL = "https://www.mcxindia.com/market-data/bhavcopy/GetDateWiseBhavCopy"

    def __init__(self):
        # Impersonate modern Chrome browser to bypass MCX TLS fingerprinting
        self.session = requests.Session(impersonate="chrome")
        self.xsrf_token = None

    def init_session(self):
        """Connects to the Bhavcopy page to collect exchange cookies and CSRF verification token."""
        print("[INFO] Initializing exchange session...")
        resp = self.session.get(self.BASE_URL, timeout=25)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        token_input = soup.find("input", {"name": "__RequestVerificationToken"})
        if token_input and token_input.get("value"):
            self.xsrf_token = token_input["value"]
        else:
            match = re.search(r'__RequestVerificationToken["\']?\s*[:=]\s*["\']([^"\']+)["\']', resp.text)
            if match:
                self.xsrf_token = match.group(1)
        print(f"[INFO] Session active. Verification token acquired: {bool(self.xsrf_token)}")

    def fetch_day(self, target_date: datetime) -> pd.DataFrame:
        """
        Fetches all contracts for a single target date.
        Handles MCX formatting quirks:
        - Input request uses DD/MM/YYYY
        - Response Date uses MM/DD/YYYY or DD MMM YYYY
        - Symbol strings contain trailing whitespace
        """
        req_date_str = target_date.strftime("%d/%m/%Y")
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": self.BASE_URL,
            "X-Requested-With": "XMLHttpRequest",
        }
        if self.xsrf_token:
            headers["__RequestVerificationToken"] = self.xsrf_token

        params = {"InstrumentName": "FUTCOM", "fromDate": req_date_str}
        resp = self.session.get(self.API_URL, params=params, headers=headers, timeout=25)

        if resp.status_code != 200:
            # Fallback to alternate internal POST endpoint if GET fails
            alt_url = "https://www.mcxindia.com/backpage.aspx/GetDateWiseBhavCopy"
            payload = {"Date": target_date.strftime("%Y%m%d"), "InstrumentName": "FUTCOM"}
            headers["Content-Type"] = "application/json; charset=utf-8"
            resp = self.session.post(alt_url, json=payload, headers=headers, timeout=25)
            resp.raise_for_status()

        data_json = resp.json()
        rows = data_json.get("Data") or data_json.get("d", {}).get("Data") or []
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.columns = df.columns.str.strip()

        # Clean space-padded commodity/symbol column
        symbol_col = "Symbol" if "Symbol" in df.columns else "Commodity"
        df["Symbol"] = df[symbol_col].astype(str).str.strip()

        # Filter strictly for the four problem statement gold contracts
        df = df[df["Symbol"].isin(CONTRACT_SPECS.keys())].copy()
        if df.empty:
            return pd.DataFrame()

        # Validate exchange returned date against requested date to filter holiday fallbacks
        df["Date_Parsed"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="mixed")
        df = df[df["Date_Parsed"].dt.date == target_date.date()].copy()
        if df.empty:
            return pd.DataFrame()

        # Standardize expiry dates
        exp_col = "ExpiryDate" if "ExpiryDate" in df.columns else "Expiry Date"
        df["Expiry_Date"] = pd.to_datetime(df[exp_col].astype(str).str.strip(), format="%d%b%Y")

        # Convert numeric pricing and volume columns
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
        df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors="coerce").fillna(0).astype(int)

        # Normalize price to 10g of 999 purity basis
        df["Quoted_Grams"] = df["Symbol"].map(lambda s: CONTRACT_SPECS[s]["quoted_grams"])
        df["Purity"] = df["Symbol"].map(lambda s: CONTRACT_SPECS[s]["purity"])
        df["Normalized_Price_10g_999"] = (
            df["Close"] * (10.0 / df["Quoted_Grams"]) * (999.0 / df["Purity"])
        ).round(2)

        df["Days_To_Expiry"] = (df["Expiry_Date"] - df["Date_Parsed"]).dt.days
        df["Contract_ID"] = df["Symbol"] + "_" + df["Expiry_Date"].dt.strftime("%Y-%m-%d")

        # Keep structured output columns
        output_cols = [
            "Contract_ID", "Symbol", "Date_Parsed", "Expiry_Date", "Days_To_Expiry",
            "Close", "Normalized_Price_10g_999", "Volume", "OpenInterest",
            "Quoted_Grams", "Purity"
        ]
        return df[output_cols]

    def scrape_range(self, start_date_str: str, end_date_str: str, output_csv="mcx_gold_2years.csv"):
        """
        Scrapes historical data over a date window with safe checkpointing.
        Saves each trading day immediately to disk and resumes automatically if interrupted.
        """
        self.init_session()
        start = datetime.strptime(start_date_str, "%Y-%m-%d")
        end = datetime.strptime(end_date_str, "%Y-%m-%d")

        # Check existing checkpoint file
        completed_dates = set()
        file_exists = os.path.exists(output_csv)
        if file_exists:
            try:
                existing_df = pd.read_csv(output_csv, usecols=["Date_Parsed"])
                completed_dates = set(pd.to_datetime(existing_df["Date_Parsed"]).dt.strftime("%Y-%m-%d").unique())
                print(f"[CHECKPOINT] Found existing data: {len(completed_dates)} trading days already saved.")
            except Exception as e:
                print(f"[CHECKPOINT] Could not read existing file: {e}. Starting fresh.")
                file_exists = False

        curr = start
        total_days = (end - start).days + 1
        processed = 0

        print(f"[INFO] Target date range: {start_date_str} to {end_date_str} (~{total_days} calendar days)")

        while curr <= end:
            curr_str = curr.strftime("%Y-%m-%d")
            processed += 1

            # Skip weekends (Saturday=5, Sunday=6)
            if curr.weekday() in (5, 6):
                curr += timedelta(days=1)
                continue

            # Skip already downloaded days
            if curr_str in completed_dates:
                curr += timedelta(days=1)
                continue

            try:
                day_df = self.fetch_day(curr)
                if not day_df.empty:
                    # Append immediately to CSV (writes header only on the very first row)
                    day_df.to_csv(output_csv, mode="a", header=not file_exists, index=False)
                    file_exists = True
                    completed_dates.add(curr_str)
                    print(f"[{processed}/{total_days}] [SAVED] {curr_str}: {len(day_df)} gold contracts saved.")
                else:
                    print(f"[{processed}/{total_days}] [SKIP]  {curr_str}: Exchange closed / holiday.")
            except Exception as e:
                print(f"[{processed}/{total_days}] [RETRY] {curr_str} error: {e}")
                time.sleep(2)

            # Polite delay to prevent rate-limiting
            time.sleep(0.6)
            curr += timedelta(days=1)

        print(f"\n[DONE] Scraping complete! Checkpoint file saved at: {output_csv}")
        if os.path.exists(output_csv):
            final_df = pd.read_csv(output_csv)
            print(f"Total historical rows: {len(final_df)} across {final_df['Date_Parsed'].nunique()} trading days.")
            return final_df
        return pd.DataFrame()


if __name__ == "__main__":
    scraper = MCXBhavcopyScraper()
    
    # 2-Year Date Range (October 2024 to October 2026)
    start_date = "2024-10-21"
    end_date = "2026-10-21"
    
    df_2yr = scraper.scrape_range(
        start_date_str=start_date,
        end_date_str=end_date,
        output_csv="mcx_gold_2years.csv"
    )
