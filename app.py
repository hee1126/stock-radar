import streamlit as st
import pandas as pd
import numpy as np

import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed


@st.cache_data(ttl=3600)
def load_krx_stocks():
    """국내 상장 종목 목록을 가져옵니다."""
    stocks = fdr.StockListing("KRX")
    stocks["Code"] = stocks["Code"].astype(str).str.zfill(6)
    stocks = stocks.dropna(subset=["Code", "Name"])
    return stocks[["Code", "Name", "Market"]].drop_duplicates()


@st.cache_data(ttl=1800)
def load_real_stock(code, name, start_date):
    """종목의 과거 일별 시세를 가져옵니다."""
    prices = fdr.DataReader(code, start_date)

    if prices is None or prices.empty:
        return pd.DataFrame()

    prices = prices.reset_index()
    prices = prices.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    prices["ticker"] = str(code).zfill(6)
    prices["name"] = name

    # 거래대금의 정확한 제공값이 없을 때 사용하는 추정치
    prices["value"] = prices["close"] * prices["volume"]

    return prices

st.set_page_config(
    page_title="주식 레이더 V4",
    page_icon="📡",
    layout="wide"
)

st.title("📡 주식 레이더")
st.caption("국내 주식 검색기 · 기술적 분석 · 과거 성과 검증 · 병렬 시세 조회")

st.warning(
    "투자 참고용 실험 도구입니다. "
    "데모 데이터는 가상 데이터이며 실제 시세가 아닙니다."
)


@st.cache_data
def make_demo_data():
    """화면 테스트용 가상 주가를 생성합니다."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                           periods=180)

    stocks = {
        "삼성전자": "005930",
        "SK하이닉스": "000660",
        "현대차": "005380",
        "가상종목A": "900001",
        "가상종목B": "900002",
    }

    frames = []

    for name, ticker in stocks.items():
        price = 50000 if name == "삼성전자" else 100000
        if name == "SK하이닉스":
            price = 200000
        if name == "현대차":
            price = 150000
        if name.startswith("가상"):
            price = 10000

        returns = rng.normal(0.0005, 0.022, len(dates))
        # 일부 구간에 임의의 상승 움직임을 넣어 테스트합니다.
        returns[100:106] += 0.025
        close = price * np.cumprod(1 + returns)
        volume = rng.integers(100000, 1500000, len(dates))

        for i, date in enumerate(dates):
            frames.append({
                "date": date,
                "ticker": ticker,
                "name": name,
                "open": close[i] * (1 + rng.normal(0, 0.005)),
                "high": close[i] * 1.012,
                "low": close[i] * 0.988,
                "close": close[i],
                "volume": volume[i],
            })

    return pd.DataFrame(frames)


def prepare_data(df):
    df = df.copy()

    required = {"date", "ticker", "name", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "필수 열이 없습니다: " + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close", "volume"])
    df = df[df["close"] > 0]
    df = df[df["volume"] >= 0]
    df = df.sort_values(["ticker", "date"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")

    # open은 다음 거래일 진입 가격에 사용합니다. CSV에 없어도 앱은 실행됩니다.
    if "open" in df.columns:
        df["open"] = pd.to_numeric(df["open"], errors="coerce")

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # 거래대금이 비어 있는 행은 종가×거래량으로 추정합니다.
        df["value"] = df["value"].fillna(df["close"] * df["volume"])
    else:
        # 거래대금 실제 값이 없으면 종가×거래량으로 추정합니다.
        df["value"] = df["close"] * df["volume"]

    parts = []

    for ticker, group in df.groupby("ticker", sort=False):
        g = group.copy()
        g["ma5"] = g["close"].rolling(5).mean()
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma60"] = g["close"].rolling(60).mean()
        # 당일을 제외한 직전 20거래일 평균을 기준으로 사용합니다.
        g["vol20"] = g["volume"].shift(1).rolling(20).mean()
        g["value20"] = g["value"].shift(1).rolling(20).mean()
        g["high20"] = g["close"].shift(1).rolling(20).max()

        # 검색기의 기본 조건과 사용자가 정한 거래량 배수 조건을 분리합니다.
        # 이렇게 해야 슬라이더 값이 검색과 검증 양쪽에 똑같이 적용됩니다.
        g["volume_ratio"] = g["volume"] / g["vol20"]
        g["value_ratio"] = g["value"] / g["value20"]
        g["signal_A"] = g["close"] > g["ma5"]
        g["signal_B"] = (
            (g["close"] > g["ma20"])
            & (g["ma5"] > g["ma20"])
            & (g["ma20"] > g["ma60"])
            & (g["close"] > g["high20"])
        )
        g["signal_C"] = (
            (g["ma5"] > g["ma20"])
            & (g["close"] > g["ma20"])
        )
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


with st.sidebar:
    st.header("데이터")
    source = st.radio(
        "데이터 선택",
        ["가상 데모 데이터", "실제 국내 주식 데이터", "내 시세 CSV 파일"]
    )

    uploaded = None
    selected_codes = []
    start_date = "2023-01-01"

    if source == "내 시세 CSV 파일":
        uploaded = st.file_uploader(
            "일별 시세 CSV 업로드",
            type=["csv"]
        )

    elif source == "실제 국내 주식 데이터":
        try:
            stock_list = load_krx_stocks()

            stock_names = dict(
                zip(stock_list["Code"], stock_list["Name"])
            )

            all_codes = list(stock_names.keys())
            analyze_all = st.checkbox(
                "전체 종목 분석",
                value=False,
                help="종목 목록에 있는 모든 코드를 조회합니다. 종목 수가 많으면 오래 걸릴 수 있습니다."
            )

            if analyze_all:
                selected_codes = all_codes
                st.caption(f"전체 분석 대상: {len(selected_codes):,}개 종목")
            else:
                defaults = [
                    code for code in ["005930", "000660", "005380"]
                    if code in stock_names
                ]
                selected_codes = st.multiselect(
                    "분석할 종목 선택",
                    options=all_codes,
                    default=defaults,
                    format_func=lambda code:
                        f"{stock_names[code]} ({code})"
                )
                st.caption(f"선택한 종목: {len(selected_codes):,}개")

            years = st.selectbox(
                "과거 데이터 기간",
                ["1년", "2년", "3년"],
                index=1
            )

            start_date = {
                "1년": "2025-01-01",
                "2년": "2024-01-01",
                "3년": "2023-01-01",
            }[years]

        except Exception as e:
            st.error(f"종목 목록을 가져오지 못했습니다: {e}")

    st.divider()
    st.header("검색기 조건")

    strategy = st.selectbox(
        "검색기",
        ["A · 거래량 급증", "B · 추세 돌파", "C · 거래량+추세"]
    )

    volume_threshold = st.slider(
        "거래량 / 20일 평균",
        1.0, 5.0, 2.0, 0.5
    )

    min_value = st.number_input(
        "최소 거래대금 (억원)",
        min_value=0,
        max_value=10000,
        value=10,
        step=10
    )

    holding_days = st.selectbox(
        "성과 확인 기간 (거래일)",
        [1, 3, 5, 10],
        index=2
    )

    round_trip_cost_pct = st.number_input(
        "왕복 거래 비용 추정치 (%)",
        min_value=0.0,
        max_value=3.0,
        value=0.3,
        step=0.05,
        help="매수·매도 수수료, 세금, 슬리피지를 합친 사용자의 추정치입니다. 실제 비용에 맞게 조정하세요."
    )

try:
    if source == "가상 데모 데이터":
        raw = make_demo_data()
        is_demo = True

    elif source == "내 시세 CSV 파일":
        if uploaded is None:
            st.info("먼저 CSV 파일을 올려 주세요.")
            st.stop()

        raw = pd.read_csv(uploaded)
        is_demo = False

    else:
        if not selected_codes:
            st.info("분석할 국내 주식을 선택해 주세요.")
            st.stop()

        stock_names = dict(
            zip(stock_list["Code"], stock_list["Name"])
        )

        frames = []
        failed_codes = []
        progress = st.progress(0)
        status = st.empty()

        # 종목을 하나씩 조회하는 대신 제한된 수의 작업을 병렬 실행합니다.
        # 너무 많은 동시 요청은 데이터 제공처의 제한을 유발할 수 있어 6개로 제한합니다.
        max_workers = min(6, len(selected_codes))

        def fetch_one(code):
            name = stock_names[code]
            try:
                prices = load_real_stock(code, name, start_date)
                if prices is None or prices.empty:
                    return code, name, None, "시세 데이터 없음"
                return code, name, prices, None
            except Exception as exc:
                return code, name, None, str(exc)[:160]

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_one, code) for code in selected_codes]
            for future in as_completed(futures):
                code, name, prices, error = future.result()
                completed += 1
                if prices is not None:
                    frames.append(prices)
                else:
                    failed_codes.append((code, name, error or "알 수 없는 오류"))

                status.write(
                    f"시세 조회 중: {completed:,}/{len(selected_codes):,} 완료 · "
                    f"최근 완료 종목: {name} ({code})"
                )
                progress.progress(completed / len(selected_codes))

        progress.empty()
        status.empty()
        st.session_state["stock_radar_failed_codes"] = failed_codes
        if failed_codes:
            st.warning(
                f"{len(failed_codes):,}개 종목의 시세를 가져오지 못했습니다. "
                "데이터 안내 탭에서 실패 종목을 확인할 수 있습니다."
            )

        if not frames:
            st.error(
                "시세를 가져오지 못했습니다. "
                "잠시 후 다시 시도하거나 다른 종목을 선택해 주세요."
            )
            st.stop()

        raw = pd.concat(frames, ignore_index=True)
        is_demo = False

    data = prepare_data(raw)

except Exception as e:
    st.error(f"데이터를 읽지 못했습니다: {e}")
    st.stop()


st.caption(
    "데모 데이터 사용 중 — 실제 주식 시세가 아닙니다."
    if is_demo else
    "업로드한 파일을 분석 중입니다. "
    "데이터 제공처와 거래대금 단위를 확인하세요."
)

# 종목별 가장 최근에 확보된 거래일을 기준으로 검색합니다.
latest = (
    data.sort_values(["ticker", "date"])
        .groupby("ticker", sort=False, as_index=False)
        .tail(1)
        .copy()
)
latest_date = latest["date"].max()

signal_col = {
    "A · 거래량 급증": "signal_A",
    "B · 추세 돌파": "signal_B",
    "C · 거래량+추세": "signal_C",
}[strategy]

# 사용자가 설정한 거래량 조건을 반영합니다.
latest["passed"] = (
    latest[signal_col]
    & (latest["volume_ratio"] >= volume_threshold)
    & (latest["value"] >= min_value * 100_000_000)
)

c1, c2, c3 = st.columns(3)
c1.metric("분석 기준일", latest_date.strftime("%Y-%m-%d"))
c2.metric("분석 종목 수", f"{latest['ticker'].nunique():,}")
c3.metric("검색 종목 수", f"{int(latest['passed'].sum()):,}")

tab1, tab2, tab3 = st.tabs(
    ["🔎 종목 검색기", "📊 성과 검증", "🗂️ 데이터 안내"]
)

with tab1:
    st.subheader("검색 결과")
    results = latest[latest["passed"]].copy()
    results["거래대금(억원)"] = (
        results["value"] / 100_000_000
    ).round(1)
    results["거래량 배수"] = results["volume_ratio"].round(2)

    show_cols = [
        "name", "ticker", "close", "ma5", "ma20", "ma60",
        "거래량 배수", "거래대금(억원)"
    ]
    results = results.rename(columns={
        "name": "종목명",
        "ticker": "종목코드",
        "date": "기준일",
        "close": "종가",
        "ma5": "5일선",
        "ma20": "20일선",
        "ma60": "60일선",
    })

    if results.empty:
        st.info("현재 조건을 만족하는 종목이 없습니다.")
    else:
        st.dataframe(
            results[[
                "종목명", "종목코드", "기준일", "종가", "5일선",
                "20일선", "60일선", "거래량 배수",
                "거래대금(억원)"
            ]],
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "검색 결과 CSV 다운로드",
            results.to_csv(index=False).encode("utf-8-sig"),
            file_name="stock_radar_results.csv",
            mime="text/csv"
        )

    selected = st.selectbox(
        "상세 확인할 종목",
        latest["name"].drop_duplicates().tolist()
    )
    hist = data[data["name"] == selected].sort_values("date")

    st.line_chart(
        hist.set_index("date")[["close", "ma5", "ma20", "ma60"]]
    )
    st.caption(
        "이동평균선은 과거 종가의 평균입니다. "
        "지지·저항선은 아직 자동 판정하지 않습니다."
    )

with tab2:
    st.subheader("검색 신호 이후의 과거 수익률")
    st.caption(
        "신호가 발생한 다음 거래일의 시가를 진입 가격으로 사용하고, "
        "설정한 보유 거래일 수에 맞춰 청산합니다. "
        "왕복 거래 비용은 입력한 추정치를 단순 차감합니다."
    )

    if "open" not in data.columns or data["open"].isna().all():
        st.warning(
            "데이터에 시가(open)가 없어 다음 거래일 종가를 진입 가격으로 사용합니다. "
            "보다 현실적인 검증을 위해 CSV에 open 열을 포함해 주세요."
        )
        use_open_price = False
    else:
        st.info(
            f"왕복 거래 비용 추정치: {round_trip_cost_pct:.2f}% "
            "(실제 증권사 수수료·세금·체결 오차와 다를 수 있습니다.)"
        )
        use_open_price = True

    rows = []
    min_value_won = min_value * 100_000_000
    cost_decimal = round_trip_cost_pct / 100

    # 검색 화면과 동일한 신호·거래량 배수·거래대금 조건을 사용합니다.
    for ticker, g in data.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)

        # 진입은 다음 거래일, 청산은 진입일을 포함한 보유 기간의 마지막 날 종가입니다.
        for i in range(max(0, len(g) - holding_days)):
            row = g.iloc[i]

            if not bool(row[signal_col]):
                continue
            if pd.isna(row["volume_ratio"]) or row["volume_ratio"] < volume_threshold:
                continue
            if pd.isna(row["value"]) or row["value"] < min_value_won:
                continue

            entry_i = i + 1
            exit_i = entry_i + holding_days - 1
            if exit_i >= len(g):
                continue

            if use_open_price:
                entry = g.loc[entry_i, "open"]
                # 다음 거래일 시가가 누락된 경우 해당 신호는 제외합니다.
                if pd.isna(entry) or entry <= 0:
                    continue
            else:
                entry = g.loc[entry_i, "close"]

            exit_price = g.loc[exit_i, "close"]
            if pd.isna(entry) or pd.isna(exit_price) or entry <= 0:
                continue

            gross_return = exit_price / entry - 1
            net_return = gross_return - cost_decimal

            rows.append({
                "종목명": row["name"],
                "종목코드": row["ticker"],
                "신호일": row["date"],
                "진입일": g.loc[entry_i, "date"],
                "청산일": g.loc[exit_i, "date"],
                "진입가": entry,
                "청산가": exit_price,
                "수익률(비용 전, %)": gross_return * 100,
                "수익률(비용 후, %)": net_return * 100,
            })

    bt = pd.DataFrame(rows)

    if bt.empty:
        st.info(
            "검증할 신호가 없습니다. 거래량 배수나 최소 거래대금을 낮추거나, "
            "더 많은 과거 데이터를 사용해 보세요."
        )
    else:
        returns = bt["수익률(비용 후, %)"].astype(float)
        avg = returns.mean()
        median = returns.median()
        win = (returns > 0).mean() * 100
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns < 0].mean()
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0:
            payoff = avg_win / abs(avg_loss)
            payoff_text = f"{payoff:.2f}"
        else:
            payoff_text = "계산 불가"
        expectancy = avg

        a, b, c = st.columns(3)
        a.metric("검증 거래 수", f"{len(bt):,}")
        b.metric("평균 수익률(비용 후)", f"{avg:.2f}%")
        c.metric("승률(비용 후)", f"{win:.1f}%")

        d, e, f = st.columns(3)
        d.metric("중앙값 수익률", f"{median:.2f}%")
        e.metric("평균 이익 거래", f"{avg_win:.2f}%" if pd.notna(avg_win) else "해당 없음")
        f.metric("평균 손실 거래", f"{avg_loss:.2f}%" if pd.notna(avg_loss) else "해당 없음")

        st.write(f"손익비(평균 이익 ÷ 평균 손실 절댓값): **{payoff_text}**")
        st.write(f"거래당 평균 기대수익(비용 후): **{expectancy:.2f}%**")
        st.caption(
            "주의: 이는 각 신호의 단순 수익률 통계입니다. 여러 종목의 동시 보유, "
            "포트폴리오 비중, 손절 규칙, 실제 세금·수수료, 시장 대비 성과는 아직 반영하지 않았습니다. "
            "같은 종목에서 보유 기간이 겹치는 신호도 별도 거래로 집계됩니다."
        )
        st.dataframe(bt, use_container_width=True, hide_index=True)

        st.download_button(
            "검증 결과 CSV 다운로드",
            bt.to_csv(index=False).encode("utf-8-sig"),
            file_name="stock_radar_backtest.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("CSV 파일 형식")
    st.write(
        "필수 열: date, ticker, name, close, volume"
    )
    st.write(
        "선택 열: open, high, low, value"
    )
    st.code(
        "date,ticker,name,close,volume,value\n"
        "2026-01-02,005930,삼성전자,100000,500000,50000000000\n"
        "2026-01-05,005930,삼성전자,101000,600000,60600000000",
        language="csv"
    )
    st.warning(
        "위 예시는 형식을 설명하기 위한 가상 숫자입니다. "
        "실제 주가가 아닙니다. 최소 60거래일 이상, "
        "가능하면 수백 거래일의 실제 일별 데이터가 필요합니다."
    )
    st.write(
        "주의: 거래대금(value)은 원 단위로 입력해야 합니다. "
        "거래대금 열이 없으면 종가×거래량으로 추정합니다."
    )
    failed_codes = st.session_state.get("stock_radar_failed_codes", [])
    if failed_codes:
        st.subheader("시세 조회 실패 종목")
        failed_df = pd.DataFrame(
            failed_codes, columns=["종목코드", "종목명", "사유"]
        )
        st.dataframe(failed_df, use_container_width=True, hide_index=True)
