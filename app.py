import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="주식 레이더",
    page_icon="📡",
    layout="wide"
)

st.title("📡 주식 레이더")
st.caption("국내 주식 검색기 · 기술적 분석 · 과거 성과 검증")

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

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        # 거래대금 실제 값이 없으면 종가×거래량으로 추정합니다.
        df["value"] = df["close"] * df["volume"]

    parts = []

    for ticker, group in df.groupby("ticker", sort=False):
        g = group.copy()
        g["ma5"] = g["close"].rolling(5).mean()
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma60"] = g["close"].rolling(60).mean()
        g["vol20"] = g["volume"].rolling(20).mean()
        g["value20"] = g["value"].rolling(20).mean()
        g["high20"] = g["close"].shift(1).rolling(20).max()

        # 과거 데이터만으로 신호를 계산합니다.
        g["volume_ratio"] = g["volume"] / g["vol20"]
        g["value_ratio"] = g["value"] / g["value20"]
        g["signal_A"] = (
            (g["volume_ratio"] >= 2)
            & (g["close"] > g["ma5"])
        )
        g["signal_B"] = (
            (g["close"] > g["ma20"])
            & (g["ma5"] > g["ma20"])
            & (g["ma20"] > g["ma60"])
            & (g["close"] > g["high20"])
        )
        g["signal_C"] = g["signal_A"] & (
            (g["ma5"] > g["ma20"])
            & (g["close"] > g["ma20"])
        )
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


with st.sidebar:
    st.header("데이터")
    source = st.radio(
        "데이터 선택",
        ["가상 데모 데이터", "내 시세 CSV 파일"]
    )
    uploaded = None

    if source == "내 시세 CSV 파일":
        uploaded = st.file_uploader(
            "일별 시세 CSV 업로드",
            type=["csv"]
        )

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


try:
    if source == "가상 데모 데이터":
        raw = make_demo_data()
        is_demo = True
    elif uploaded is not None:
        raw = pd.read_csv(uploaded)
        is_demo = False
    else:
        st.info(
            "왼쪽에서 CSV 파일을 올리거나 "
            "가상 데모 데이터를 선택해 주세요."
        )
        st.stop()

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

latest_date = data["date"].max()
latest = data[data["date"] == latest_date].copy()

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
                "종목명", "종목코드", "종가", "5일선",
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
        "신호가 발생한 종가에 매수했다고 가정한 단순 계산입니다. "
        "실제 체결 가능 가격, 수수료, 세금, 슬리피지는 "
        "반영하지 않습니다."
    )

    rows = []

    # 검색 화면과 동일한 조건으로 과거 신호를 검증합니다.
    min_value_won = min_value * 100_000_000

    for ticker, g in data.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)

        for i in range(len(g) - holding_days - 1):
            row = g.iloc[i]

            # 선택한 검색기의 기본 조건
            if not bool(row[signal_col]):
                continue

            # 화면에서 설정한 거래량 조건
            if pd.isna(row["volume_ratio"]):
                continue
            if row["volume_ratio"] < volume_threshold:
                continue

            # 화면에서 설정한 최소 거래대금 조건
            if pd.isna(row["value"]) or row["value"] < min_value_won:
                continue

            # 신호 다음 거래일 종가를 진입 가격으로 가정
            entry_i = i + 1
            exit_i = entry_i + holding_days

            entry = g.loc[entry_i, "close"]
            exit_price = g.loc[exit_i, "close"]

            if pd.isna(entry) or pd.isna(exit_price) or entry <= 0:
                continue

            rows.append({
                "종목명": row["name"],
                "신호일": row["date"],
                "진입일": g.loc[entry_i, "date"],
                "청산일": g.loc[exit_i, "date"],
                "진입가": entry,
                "청산가": exit_price,
                "수익률(%)": (exit_price / entry - 1) * 100,
            })

    bt = pd.DataFrame(rows)

    if bt.empty:
        st.info(
            "검증할 신호가 충분하지 않습니다. "
            "더 많은 과거 데이터를 업로드해 주세요."
        )
    else:
        avg = bt["수익률(%)"].mean()
        win = (bt["수익률(%)"] > 0).mean() * 100
        median = bt["수익률(%)"].median()

        a, b, c = st.columns(3)
        a.metric("신호 수", f"{len(bt):,}")
        b.metric("평균 수익률", f"{avg:.2f}%")
        c.metric("양수 수익률 비율", f"{win:.1f}%")

        st.write(f"중앙값 수익률: **{median:.2f}%**")
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
