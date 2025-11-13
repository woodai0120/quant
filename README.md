# Efficiency Growth Screener

This repository contains an API-driven implementation of the Efficiency Growth stock screener for Korean equities. The script pulls price data from **pykrx** and fundamental data from the **DART** API, then evaluates each security against six rules and exports CSV reports.

## Prerequisites

1. **Python** 3.9 or later.
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain a DART API key from <https://opendart.fss.or.kr/> and create a `.env` file in the project root with at least:

   ```bash
   DART_API_KEY=your_api_key_here
   ```

   Optionally, set `MIN_PASS_POINTS` to relax the minimum number of conditions a company must satisfy (default: `6`).

## Running the screener

```bash
python efficiency_growth_api.py
```

The script performs a quick smoke test (Samsung Electronics: `005930`) and then scans the selected market universe. Two CSV files are produced in the repository root:

- `kr_screen_result.csv`: full screener output.
- `kr_screen_result_top20.csv`: top-20 market-cap names among the passing companies.

### Configuration flags

Adjust the module-level constants near the top of `efficiency_growth_api.py` to control runtime behaviour:

- `UNIVERSE_MODE`: Universe selection (`"KOSPI200"`, `"KOSPI"`, or `"ALL"`).
- `FAST_MODE`: When `True`, only the largest `FAST_LIMIT` tickers are processed first.
- `FAST_LIMIT`: Number of tickers to evaluate in fast mode.

### Notes

- Each execution makes multiple API calls to DART. Avoid running the screener too frequently to respect rate limits.
- The script caches DART responses during a single run, but it does not persist data between runs.
- Financial sector tickers are automatically excluded from PSR and PRR checks, matching the original screening logic.

## Troubleshooting

- If you see `❌ .env에 DART_API_KEY가 없습니다.`, the environment variable is missing. Verify your `.env` file and ensure it sits in the repository root.
- Network hiccups from DART are retried automatically. Persistent `status` errors (e.g., `013`, `020`) usually indicate quota or authentication issues.
