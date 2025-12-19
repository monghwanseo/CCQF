import gzip
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
OUT_DIR = Path("data_set")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_KST = datetime(2020, 1, 1, 0, 0, 0)
END_KST = datetime(2024, 12, 31, 23, 59, 0)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000


def kst_to_utc_ms(dt_kst: datetime) -> int:
    kst = timezone(timedelta(hours=9))
    dt = dt_kst.replace(tzinfo=kst)
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def utc_ms_to_kst_str(ms: int) -> str:
    dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    dt_kst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return dt_kst.strftime("%Y-%m-%d %H:%M:%S")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        all_rows.extend(data)

        last_open_time = data[-1][0]
        next_ms = last_open_time + 60_000
        if next_ms <= cur:
            break
        cur = next_ms

        time.sleep(0.2)
    return all_rows


def main():
    start_ms = kst_to_utc_ms(START_KST)
    end_ms = kst_to_utc_ms(END_KST + timedelta(minutes=1))

    rows = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)
    if not rows:
        raise SystemExit("No data fetched from Binance.")

    records = []
    for r in rows:
        open_time = int(r[0])
        open_price = float(r[1])
        high_price = float(r[2])
        low_price = float(r[3])
        close_price = float(r[4])
        volume = float(r[5])
        time_kst = utc_ms_to_kst_str(open_time)
        records.append(
            (time_kst, open_price, high_price, low_price, close_price, volume)
        )

    df = pd.DataFrame(
        records,
        columns=["time_kst", "open", "high", "low", "close", "volume"],
    )
    df.drop_duplicates(subset=["time_kst"], keep="last", inplace=True)
    df.sort_values("time_kst", inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = OUT_DIR / "BTC_1m.csv.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    print(f"[save] {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
