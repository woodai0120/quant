"""Efficiency Growth (API-only) screener implementation.

This module provides the "Efficiency Growth" stock screener logic that relies
exclusively on public APIs.  It evaluates Korean equities against the following
six conditions:

1. Positive three-year average net profit margin.
2. Price-to-R&D ratio (PRR) ≤ 15 (financial sector automatically excluded).
3. Price-to-sales ratio (PSR) ≤ 0.5 (financial sector automatically excluded).
4. R&D-to-assets ratio (RAR) between 0 and 50%.
5. Accelerating revenue YoY growth (latest quarter YoY > previous quarter YoY).
6. Cash-to-market-capitalisation ratio ≥ 30%.

Running the script will produce two CSV files:

* ``kr_screen_result.csv`` – full screener output for all processed tickers.
* ``kr_screen_result_top20.csv`` – top 20 tickers by market cap among the
  passing securities.

The logic is largely derived from the original notebook version but has been
adapted so it can run headless inside the repository.
"""

from __future__ import annotations

import io
import os
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Third-party data source that provides market information for Korean stocks.
from pykrx import stock


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

# The screener relies on the DART API.  The API key must be provided via a
# ``.env`` file placed in the project root.
load_dotenv()
DART_API_KEY = os.getenv("DART_API_KEY")
if not DART_API_KEY:
    raise ValueError("❌ .env에 DART_API_KEY가 없습니다.")

# ``MIN_PASS_POINTS`` can be used to relax the number of conditions that must
# be satisfied before a stock is considered to have passed the screener.
MIN_PASS_POINTS = int(os.getenv("MIN_PASS_POINTS", "6"))


# ---------------------------------------------------------------------------
# Screener configuration
# ---------------------------------------------------------------------------

UNIVERSE_MODE = "KOSPI200"  # Options: "KOSPI200", "KOSPI", "ALL"
FAST_MODE = True  # Whether to only process the top market-cap names first
FAST_LIMIT = 30
OUT_ALL = "kr_screen_result.csv"
OUT_TOP = "kr_screen_result_top20.csv"

REQ_TIMEOUT = (5, 20)  # ``requests`` connect & read timeout
PAUSE_DART = 0.25  # Sleep between DART requests to respect rate limits
FS_DIV_ORDER = ("CFS", "OFS")  # Prefer consolidated financial statements


# ---------------------------------------------------------------------------
# HTTP session helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """Create a configured :class:`requests.Session` instance."""

    session = requests.Session()
    session.trust_env = False

    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0) KR-Screener"})
    return session


SESSION = make_session()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pkg_resources")


# ---------------------------------------------------------------------------
# KRX helpers
# ---------------------------------------------------------------------------


def normalize_code(code: str) -> str:
    return str(code).strip().zfill(6)


def latest_bday(max_lookback: int = 7) -> str:
    today = datetime.today()
    for i in range(max_lookback):
        date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            df = stock.get_market_cap_by_ticker(date_str, market="KOSPI")
        except Exception:
            continue
        if df is not None and len(df) > 10:
            return date_str
    return today.strftime("%Y%m%d")


def get_universe(mode: str) -> List[str]:
    day = latest_bday()
    if mode == "KOSPI200":
        return stock.get_index_portfolio_deposit_file("1028", day)
    if mode == "KOSPI":
        return stock.get_market_ticker_list(day, market="KOSPI")
    if mode == "ALL":
        kospi = set(stock.get_market_ticker_list(day, market="KOSPI"))
        kosdaq = set(stock.get_market_ticker_list(day, market="KOSDAQ"))
        return sorted(kospi | kosdaq)
    raise ValueError("UNIVERSE_MODE 잘못됨")


CAP_SNAPSHOT: Optional[pd.DataFrame] = None


def get_cap_snapshot() -> pd.DataFrame:
    global CAP_SNAPSHOT
    if CAP_SNAPSHOT is None:
        day = latest_bday()
        kospi = stock.get_market_cap_by_ticker(day, market="KOSPI")
        kosdaq = stock.get_market_cap_by_ticker(day, market="KOSDAQ")
        cap = pd.concat([kospi, kosdaq], axis=0)
        cap.index = cap.index.astype(str).str.zfill(6)
        CAP_SNAPSHOT = cap
    return CAP_SNAPSHOT


def get_price_block(code: str) -> Optional[Dict[str, float]]:
    code = normalize_code(code)
    cap = get_cap_snapshot()
    if code not in cap.index:
        return None
    row = cap.loc[code]
    return {
        "price": float(row.get("종가", np.nan)),
        "mcap": float(row.get("시가총액", np.nan)),
    }


# ---------------------------------------------------------------------------
# DART helpers
# ---------------------------------------------------------------------------


def dart_corp_map() -> Dict[str, str]:
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    response = SESSION.get(url, timeout=REQ_TIMEOUT)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(archive.read("CORPCODE.xml"))
        mapping = {}
        for el in root.findall("list"):
            sc = el.findtext("stock_code")
            cc = el.findtext("corp_code")
            if sc and cc:
                mapping[str(sc).strip().zfill(6)] = cc.strip()
    if not mapping:
        raise RuntimeError("DART corp map empty")
    return mapping


@lru_cache(maxsize=8192)
def _dart_fetch_tuple(corp_code: str, bsns_year: int, reprt_code: str, fs_div: str):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(bsns_year),
        "reprt_code": reprt_code,
        "fs_div": fs_div,
    }

    max_retry, pause = 4, 0.6
    last_msg = None
    for i in range(max_retry):
        try:
            response = SESSION.get(url, params=params, timeout=REQ_TIMEOUT)
            data = response.json()
            status = data.get("status")
            if status == "000" and data.get("list"):
                return tuple(data["list"])
            last_msg = data.get("message")
            if status in ("013", "020", "100", "101"):
                print(
                    f"[DART {reprt_code} {bsns_year} {fs_div}] stop status={status}, msg={last_msg}"
                )
                break
            print(
                f"[DART {reprt_code} {bsns_year} {fs_div}] retry {i + 1}/4 status={status}, msg={last_msg}"
            )
        except Exception as exc:  # noqa: BLE001 - DART occasionally fails
            last_msg = str(exc)
            if i == max_retry - 1:
                print(f"[DART EXC {reprt_code} {bsns_year} {fs_div}] {exc}")
        time.sleep(pause * (i + 1))
    return None


def dart_fs_single_try(
    corp_code: str, bsns_year: int, reprt_code: str, fs_order: Iterable[str] = FS_DIV_ORDER
):
    for fs in fs_order:
        data = _dart_fetch_tuple(corp_code, int(bsns_year), reprt_code, fs)
        if data:
            df = pd.DataFrame(list(data))
            for col in ("thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            return df, fs
    return None, None


# ---------------------------------------------------------------------------
# Label matching utilities
# ---------------------------------------------------------------------------


def _norm_labels(df: pd.DataFrame) -> pd.Series:
    return df["account_nm"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()


def pick_amount(df: Optional[pd.DataFrame], keys: Iterable[str]) -> Optional[float]:
    if df is None or "account_nm" not in df.columns:
        return None
    labels = _norm_labels(df)

    for key in keys:
        key_norm = str(key).replace(" ", "").lower()
        hit = df[labels == key_norm]
        if len(hit) > 0:
            values = pd.to_numeric(hit["thstrm_amount"], errors="coerce").dropna()
            if len(values) > 0:
                return float(values.iloc[0])

    for key in keys:
        key_norm = str(key).replace(" ", "").lower()
        mask = labels.str.contains(key_norm, regex=False)
        hit = df[mask]
        if len(hit) > 0:
            values = pd.to_numeric(hit["thstrm_amount"], errors="coerce").dropna()
            if len(values) > 0:
                return float(values.iloc[0])
    return None


# ---------------------------------------------------------------------------
# Financial data collection
# ---------------------------------------------------------------------------


def collect_financials(corp_code: str) -> Dict[str, Optional[float]]:
    keys_sales = [
        "매출액",
        "영업수익",
        "수익(매출)",
        "revenue",
        "매출",
        "매출수익",
        "이자수익",
        "보험료수익",
        "영업수익(보험)",
        "보험영업수익",
    ]
    keys_net = ["당기순이익", "순이익", "net income", "연결당기순이익", "분기순이익"]
    keys_rnd = ["연구개발", "연구개발비", "r&d", "개발비"]

    cash_exact = [
        "현금및현금성자산",
        "현금및현금성자산합계",
        "현금및 현금성자산",
        "현금 및 현금성자산",
        "현금및현금성자산등",
        "현금및현금성자산등합계",
    ]
    cash_alt_components = [
        "단기금융상품",
        "단기예치금",
        "현금및예치금",
        "현금및예치금등",
        "현금및예치금합계",
        "현금및예금",
        "예치금",
        "당좌자산중현금및예치금",
    ]

    asset_exact = ["자산총계", "총자산", "자산 합계", "자산합계", "자산합계총계"]
    debt_exact = ["부채총계", "총부채", "부채 합계", "부채합계"]
    equity_exact = ["자본총계", "총자본", "자본 합계", "자본합계", "자본과부채총계"]

    this_year = datetime.today().year

    values: List[Tuple[Optional[float], Optional[float]]] = []
    rnd_latest: Optional[float] = None
    for year in [this_year - 1, this_year - 2, this_year - 3]:
        df_year, _ = dart_fs_single_try(corp_code, year, "1104")
        time.sleep(PAUSE_DART)
        if df_year is None:
            continue
        sales = pick_amount(df_year, keys_sales)
        net_income = pick_amount(df_year, keys_net)
        rnd = pick_amount(df_year, keys_rnd)
        if rnd is not None and rnd_latest is None:
            rnd_latest = rnd
        values.append((sales, net_income))

    avg_npm = None
    if len(values) >= 3:
        ratios = [n / s for (s, n) in values[-3:] if s and n and s != 0]
        if len(ratios) == 3:
            avg_npm = float(np.mean(ratios))
    ttm_sales = values[-1][0] if len(values) > 0 else None

    def get_quarter_cash_assets(year: int) -> Tuple[Optional[float], Optional[float]]:
        for rc in ["1103", "1102", "1101"]:
            dfq, _ = dart_fs_single_try(corp_code, year, rc)
            time.sleep(PAUSE_DART / 2)
            if dfq is None:
                continue
            cash = pick_amount(dfq, cash_exact)
            if cash is None:
                pieces = []
                for key in cash_alt_components:
                    value = pick_amount(dfq, [key])
                    if value is not None:
                        pieces.append(value)
                if pieces:
                    cash = float(np.nansum(pieces))
            assets = pick_amount(dfq, asset_exact)
            if assets is None:
                debt = pick_amount(dfq, debt_exact)
                equity = pick_amount(dfq, equity_exact)
                if debt is not None and equity is not None:
                    assets = float(debt) + float(equity)
            if (cash is not None) or (assets is not None):
                return cash, assets
        return None, None

    cash, assets = get_quarter_cash_assets(this_year)

    if cash is None or assets is None:
        df_last_year, _ = dart_fs_single_try(corp_code, this_year - 1, "1104")
        time.sleep(PAUSE_DART / 2)
        if df_last_year is not None:
            if cash is None:
                c_exact = pick_amount(df_last_year, cash_exact)
                if c_exact is None:
                    pieces = []
                    for key in cash_alt_components:
                        value = pick_amount(df_last_year, [key])
                        if value is not None:
                            pieces.append(value)
                    if pieces:
                        c_exact = float(np.nansum(pieces))
                cash = c_exact
            if assets is None:
                a_exact = pick_amount(df_last_year, asset_exact)
                if a_exact is None:
                    debt = pick_amount(df_last_year, debt_exact)
                    equity = pick_amount(df_last_year, equity_exact)
                    if debt is not None and equity is not None:
                        a_exact = float(debt) + float(equity)
                assets = a_exact

    return {
        "avg_npm_3y": avg_npm,
        "ttm_sales": ttm_sales,
        "rnd": rnd_latest,
        "cash": cash,
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# Quarterly YoY acceleration
# ---------------------------------------------------------------------------


def dart_quarter_sales_series(corp_code: str):
    keys = [
        "매출액",
        "영업수익",
        "수익(매출)",
        "revenue",
        "매출",
        "매출수익",
        "이자수익",
        "보험료수익",
        "영업수익(보험)",
        "보험영업수익",
    ]
    this_year = datetime.today().year

    def quarter_sum(year: int, rc: str) -> Optional[float]:
        df, _ = dart_fs_single_try(corp_code, year, rc)
        time.sleep(PAUSE_DART / 3)
        return pick_amount(df, keys) if df is not None else None

    for base_year in [this_year, this_year - 1]:
        q1 = quarter_sum(base_year, "1101")
        h1 = quarter_sum(base_year, "1102")
        q3c = quarter_sum(base_year, "1103")
        q1p = quarter_sum(base_year - 1, "1101")
        h1p = quarter_sum(base_year - 1, "1102")
        q3p = quarter_sum(base_year - 1, "1103")

        Q1 = (q1, q1p)
        Q2 = ((h1 - q1) if (h1 and q1) else None, (h1p - q1p) if (h1p and q1p) else None)
        Q3 = ((q3c - h1) if (q3c and h1) else None, (q3p - h1p) if (q3p and h1p) else None)

        usable = {"Q3": Q3, "Q2": Q2, "Q1": Q1}
        cnt_ok = sum(int(v[0] and v[1] and v[1] != 0) for v in usable.values())
        if cnt_ok >= 2:
            return usable

    return {"Q3": (None, None), "Q2": (None, None), "Q1": (None, None)}


def calc_sales_yoy_accel(corp_code: str):
    series = dart_quarter_sales_series(corp_code)
    order = ["Q3", "Q2", "Q1"]
    usable = [q for q in order if series[q][0] and series[q][1] and series[q][1] != 0]
    if len(usable) < 2:
        return None, None, None

    def yoy(pair: Tuple[Optional[float], Optional[float]]) -> Optional[float]:
        cur, prev = pair
        return (cur - prev) / prev * 100 if prev else None

    now_quarter, prev_quarter = usable[0], usable[1]
    now = yoy(series[now_quarter])
    prev = yoy(series[prev_quarter])
    accel = (now is not None and prev is not None and now > prev)
    return accel, now, prev


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def score_company(code: str, corp_map: Dict[str, str]):
    code = normalize_code(code)
    name = stock.get_market_ticker_name(code)

    price = get_price_block(code)
    if not price or not np.isfinite(price["mcap"]):
        return None

    corp_code = corp_map.get(code)
    fin = collect_financials(corp_code) if corp_code else {}
    avg = fin.get("avg_npm_3y")
    ttm = fin.get("ttm_sales")
    rnd = fin.get("rnd")
    cash = fin.get("cash")
    assets = fin.get("assets")

    name_lower = (name or "").lower()
    is_fin = any(
        keyword in name_lower for keyword in ["은행", "금융", "지주", "보험", "화재", "생명", "증권", "캐피탈", "카드"]
    )

    psr = price["mcap"] / ttm if (ttm and ttm > 0 and not is_fin) else None
    prr = price["mcap"] / rnd if (rnd and rnd > 0 and not is_fin) else None
    rar = (rnd / assets * 100) if (rnd and assets and assets > 0) else None

    accel, now, prev = calc_sales_yoy_accel(corp_code) if corp_code else (None, None, None)

    checks = {
        "P1_npm3y_pos": (avg is not None and avg > 0),
        "P2_prr_le_15": (prr is not None and prr <= 15),
        "P3_psr_le_0p5": (psr is not None and psr <= 0.5),
        "P4_rar_0to50": (rar is not None and 0 <= rar <= 50),
        "G1_yoy_accel": (accel if accel is not None else None),
        "F1_cash_ge_30pct": (cash and price["mcap"] > 0 and (cash / price["mcap"]) >= 0.30),
    }

    usable = [k for k, v in checks.items() if v is not None]
    points = sum(1 for v in checks.values() if v is True)

    need = min(MIN_PASS_POINTS, len(usable)) if len(usable) > 0 else MIN_PASS_POINTS
    passed = points >= need

    if cash is None or assets is None:
        print(f"[WARN] 분기재무 공란(현금/자산): {code} {name}")
    if accel is None:
        print(f"[WARN] YoY 가속 불가: {code} {name}")

    return {
        "code": code,
        "name": name,
        "passed": passed,
        "points": points,
        "need": need,
        "avail": len(usable),
        "mcap": price["mcap"],
        "psr": psr,
        "prr": prr,
        "rar": rar,
        "avg_npm_3y": avg,
        "cash_to_mcap": (None if not cash else cash / price["mcap"]),
        "yoy_now": now,
        "yoy_prev": prev,
    }


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def main() -> None:
    universe = get_universe(UNIVERSE_MODE)
    print(f"[Universe] {UNIVERSE_MODE}: {len(universe)} 종목")

    if FAST_MODE:
        cap = get_cap_snapshot()
        codes = [normalize_code(c) for c in universe]
        cap_sub = cap.loc[cap.index.intersection(codes)]
        universe = cap_sub.sort_values("시가총액", ascending=False).index.tolist()[:FAST_LIMIT]
        print(f"[FAST] 제한 실행: {len(universe)} 종목(상위 시총)")

    corp_map = dart_corp_map()
    results = []
    for code in tqdm(universe, desc="Scanning"):
        try:
            result = score_company(code, corp_map)
            if result:
                results.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {code}: {exc}")
            continue

    df = pd.DataFrame(results)
    if df.empty:
        print("\n❗ 결과 없음.")
        return

    df = df.sort_values(["passed", "points", "mcap"], ascending=[False, False, False])
    df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    print(f"\n전체 저장 → {OUT_ALL}")

    passed = df[df["passed"] == True].copy()  # noqa: E712
    if passed.empty:
        print("\n❗ 통과 없음. 상위 30 미리보기:")
        cols = [
            c
            for c in [
                "code",
                "name",
                "points",
                "need",
                "avail",
                "mcap",
                "psr",
                "prr",
                "rar",
                "avg_npm_3y",
                "cash_to_mcap",
                "yoy_now",
                "yoy_prev",
            ]
            if c in df.columns
        ]
        print(df.head(30)[cols].to_string(index=False))
        return

    top20 = passed.sort_values(["mcap", "points"], ascending=[False, False]).head(20).copy()
    cols = [
        c
        for c in [
            "code",
            "name",
            "points",
            "need",
            "avail",
            "mcap",
            "psr",
            "prr",
            "rar",
            "avg_npm_3y",
            "cash_to_mcap",
            "yoy_now",
            "yoy_prev",
        ]
        if c in top20.columns
    ]
    print("\n=== 통과 종목 — 시총 Top 20 ===")
    print(top20[cols].to_string(index=False))
    top20.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")
    print(f"\nTop 20 저장 → {OUT_TOP}")


def smoke() -> None:
    corp_map = dart_corp_map()
    code = "005930"
    result = score_company(code, corp_map)
    if result is None:
        print("[SMOKE] 005930 결과 없음")
        return
    keys = [
        "code",
        "name",
        "points",
        "need",
        "avail",
        "passed",
        "mcap",
        "psr",
        "prr",
        "rar",
        "avg_npm_3y",
        "cash_to_mcap",
        "yoy_now",
        "yoy_prev",
    ]
    preview = {k: result.get(k) for k in keys}
    print("\n[SMOKE] 005930 결과 미리보기")
    print(preview)


if __name__ == "__main__":
    smoke()
    main()

