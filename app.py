import math
import heapq
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt

# =========================================================
# Arctic Green Routing Dashboard
# Streamlit prototype for logistics competition presentation
# Theme: Speed-driven Black Carbon Risk + Dynamic Arctic Routing
# =========================================================

st.set_page_config(
    page_title="Arctic Green Routing",
    page_icon="🧊",
    layout="wide"
)

# -----------------------------
# 1. Basic styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.0rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Arctic Green Routing System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time Ice-Carbon Routing Prototype: speed, ice thickness, fuel type, and black carbon risk</div>',
    unsafe_allow_html=True
)

# -----------------------------
# 2. Input panel: number-input style, no sliders
# -----------------------------
with st.sidebar:
    st.header("운항 조건 입력")

    fuel_type = st.selectbox(
        "연료 종류",
        ["HFO", "VLSFO", "LNG", "Green Methanol"],
        index=0
    )

    speed = st.number_input(
        "선속 Speed (knots)",
        min_value=6.0,
        max_value=24.0,
        value=14.0,
        step=0.5
    )

    engine_load = st.number_input(
        "엔진 부하 Engine Load (%)",
        min_value=20.0,
        max_value=100.0,
        value=70.0,
        step=5.0
    )

    ship_dwt = st.number_input(
        "선박 규모 DWT",
        min_value=10000,
        max_value=250000,
        value=80000,
        step=5000
    )

    distance_nm = st.number_input(
        "예상 항해 거리 (nautical miles)",
        min_value=1000,
        max_value=12000,
        value=5600,
        step=100
    )

    ice_multiplier = st.number_input(
        "해빙 위험 보정계수 Ice Risk Multiplier",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

    carbon_tax = st.number_input(
        "탄소비용 Carbon Price ($/tCO2eq)",
        min_value=0,
        max_value=300,
        value=85,
        step=5
    )

    st.divider()
    st.caption("출발/도착 좌표는 발표용 기본값입니다. 필요하면 직접 바꿀 수 있습니다.")

    start_lat = st.number_input("출발 위도", value=69.65, step=0.1)
    start_lon = st.number_input("출발 경도", value=33.75, step=0.5)
    end_lat = st.number_input("도착 위도", value=66.05, step=0.1)
    end_lon = st.number_input("도착 경도", value=169.70, step=0.5)

# -----------------------------
# 3. Model assumptions
# -----------------------------
FUEL_FACTORS = {
    # relative black carbon emission factor
    "HFO": 1.00,
    "VLSFO": 0.72,
    "LNG": 0.28,
    "Green Methanol": 0.12,
}

CO2_FACTORS = {
    # rough relative CO2eq factor for prototype calculation
    "HFO": 3.114,
    "VLSFO": 3.050,
    "LNG": 2.750,
    "Green Methanol": 0.850,
}

fuel_factor = FUEL_FACTORS[fuel_type]
co2_factor = CO2_FACTORS[fuel_type]

# -----------------------------
# 4. Synthetic ice-risk grid
#    Replace this later with real NSIDC / Copernicus / satellite ice data.
# -----------------------------
@st.cache_data
def generate_arctic_grid():
    lat_values = np.arange(66, 85.5, 0.75)
    lon_values = np.arange(20, 171, 2.5)
    rows = []

    for lat in lat_values:
        for lon in lon_values:
            # Synthetic ice pattern:
            # higher latitude + central Arctic corridor = thicker ice proxy
            latitude_component = max(0, (lat - 66) / 19)
            central_ice_belt = math.exp(-((lon - 95) ** 2) / (2 * 32 ** 2))
            local_variation = 0.12 * math.sin(math.radians(lon * 2.1)) + 0.08 * math.cos(math.radians(lat * 5))

            ice_thickness = 0.15 + 2.2 * latitude_component + 1.15 * central_ice_belt + local_variation
            ice_thickness = max(0.05, ice_thickness)

            rows.append({
                "lat": lat,
                "lon": lon,
                "ice_thickness": ice_thickness
            })

    return pd.DataFrame(rows)

base_grid = generate_arctic_grid()

# -----------------------------
# 5. Black carbon risk formula
# -----------------------------
def calculate_bc_risk(speed_knots, ice_thickness, fuel_factor, engine_load_pct, ice_multiplier):
    """
    Prototype formula.
    Core logic:
    - Open-water resistance roughly rises with speed^2.
    - Ice resistance rises with ice thickness and speed^1.5.
    - High engine load increases incomplete combustion risk.
    - HFO has highest BC risk.
    """
    k_open = 0.50
    k_ice = 2.50

    resistance = k_open * (speed_knots ** 2) + k_ice * ice_thickness * ice_multiplier * (speed_knots ** 1.5)
    load_multiplier = 1 + ((engine_load_pct - 50) / 100) ** 2 * 1.6
    bc_risk = resistance * fuel_factor * load_multiplier

    return bc_risk

risk_grid = base_grid.copy()
risk_grid["bc_risk"] = risk_grid["ice_thickness"].apply(
    lambda ice: calculate_bc_risk(speed, ice, fuel_factor, engine_load, ice_multiplier)
)
risk_grid["risk_norm"] = (risk_grid["bc_risk"] - risk_grid["bc_risk"].min()) / (risk_grid["bc_risk"].max() - risk_grid["bc_risk"].min())

# Color: green -> yellow -> red
risk_grid["color"] = risk_grid["risk_norm"].apply(
    lambda r: [
        int(40 + 215 * r),
        int(180 - 130 * r),
        int(80 - 50 * r),
        150
    ]
)

# -----------------------------
# 6. Dynamic path finding using grid graph + Dijkstra
# -----------------------------
def nearest_grid_point(df, lat, lon):
    dist = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    idx = dist.idxmin()
    return (df.loc[idx, "lat"], df.loc[idx, "lon"])


def build_grid_index(df):
    points = {(row.lat, row.lon): row.bc_risk for row in df.itertuples()}
    return points


def dijkstra_path(df, start, end, speed_weight=0.35, risk_weight=0.65):
    points = build_grid_index(df)
    lats = sorted(df["lat"].unique())
    lons = sorted(df["lon"].unique())
    lat_step = round(lats[1] - lats[0], 2)
    lon_step = round(lons[1] - lons[0], 2)

    start = nearest_grid_point(df, start[0], start[1])
    end = nearest_grid_point(df, end[0], end[1])

    max_risk = df["bc_risk"].max()
    min_risk = df["bc_risk"].min()

    def neighbors(node):
        lat, lon = node
        candidates = [
            (lat + lat_step, lon),
            (lat - lat_step, lon),
            (lat, lon + lon_step),
            (lat, lon - lon_step),
            (lat + lat_step, lon + lon_step),
            (lat + lat_step, lon - lon_step),
            (lat - lat_step, lon + lon_step),
            (lat - lat_step, lon - lon_step),
        ]
        return [(round(a, 2), round(b, 2)) for a, b in candidates if (round(a, 2), round(b, 2)) in points]

    pq = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while pq:
        _, current = heapq.heappop(pq)
        if current == end:
            break

        for nxt in neighbors(current):
            lat1, lon1 = current
            lat2, lon2 = nxt
            step_distance = math.sqrt((lat2 - lat1) ** 2 + ((lon2 - lon1) * 0.45) ** 2)
            normalized_risk = (points[nxt] - min_risk) / (max_risk - min_risk)
            new_cost = cost_so_far[current] + speed_weight * step_distance + risk_weight * normalized_risk

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost
                heapq.heappush(pq, (priority, nxt))
                came_from[nxt] = current

    if end not in came_from:
        return [start, end]

    path = [end]
    node = end
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path

path_points = dijkstra_path(
    risk_grid,
    start=(start_lat, start_lon),
    end=(end_lat, end_lon)
)

path_coordinates = [[lon, lat] for lat, lon in path_points]
path_df = pd.DataFrame({"path": [path_coordinates]})

# Calculate route risk along selected path
path_lookup = risk_grid.set_index(["lat", "lon"])["bc_risk"].to_dict()
path_risks = [path_lookup.get((lat, lon), risk_grid["bc_risk"].mean()) for lat, lon in path_points]
avg_path_risk = float(np.mean(path_risks))
max_path_risk = float(np.max(path_risks))

# -----------------------------
# 7. KPI calculations
# -----------------------------
avg_ice = float(risk_grid["ice_thickness"].mean())
base_resistance = 0.50 * speed ** 2 + 2.50 * avg_ice * ice_multiplier * speed ** 1.5
fuel_consumption_index = base_resistance * distance_nm * (ship_dwt / 80000) / 1000
co2eq_tons = fuel_consumption_index * co2_factor
bc_index = avg_path_risk * distance_nm / 1000
carbon_cost = co2eq_tons * carbon_tax

# Recommended speed search
candidate_speeds = np.arange(8, 21, 0.5)
recommendations = []
for spd in candidate_speeds:
    temp_risks = base_grid["ice_thickness"].apply(
        lambda ice: calculate_bc_risk(spd, ice, fuel_factor, engine_load, ice_multiplier)
    )
    avg_risk = temp_risks.mean()
    travel_time_days = distance_nm / (spd * 24)
    # Objective: balance time and black carbon risk
    score = 0.72 * (avg_risk / risk_grid["bc_risk"].mean()) + 0.28 * (travel_time_days / (distance_nm / (14 * 24)))
    recommendations.append({"speed": spd, "score": score, "avg_bc_risk": avg_risk, "days": travel_time_days})

recommend_df = pd.DataFrame(recommendations)
best_row = recommend_df.loc[recommend_df["score"].idxmin()]
recommended_speed = float(best_row["speed"])

current_bc = float(recommend_df.loc[recommend_df["speed"].sub(speed).abs().idxmin(), "avg_bc_risk"])
recommended_bc = float(best_row["avg_bc_risk"])
bc_reduction = max(0, (current_bc - recommended_bc) / current_bc * 100)

# -----------------------------
# 8. Main layout
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 선속", f"{speed:.1f} knots")
col2.metric("AI 권장 선속", f"{recommended_speed:.1f} knots")
col3.metric("예상 BC 절감률", f"{bc_reduction:.1f}%")
col4.metric("탄소비용 추정", f"${carbon_cost:,.0f}")

left, right = st.columns([1.65, 1])

with left:
    st.subheader("북극 BC Risk Heatmap + Dynamic Path")

    heatmap_layer = pdk.Layer(
        "ScatterplotLayer",
        data=risk_grid,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=42000,
        pickable=True,
        opacity=0.72,
    )

    path_layer = pdk.Layer(
        "PathLayer",
        data=path_df,
        get_path="path",
        get_width=6,
        get_color=[30, 90, 255, 230],
        width_min_pixels=4,
        pickable=True,
    )

    start_end_df = pd.DataFrame([
        {"lat": start_lat, "lon": start_lon, "label": "START", "color": [0, 120, 255, 220]},
        {"lat": end_lat, "lon": end_lon, "label": "END", "color": [20, 180, 80, 220]},
    ])

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=start_end_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=85000,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=74.0,
        longitude=95.0,
        zoom=2.25,
        pitch=35,
    )

    deck = pdk.Deck(
        layers=[heatmap_layer, path_layer, point_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>BC Risk:</b> {bc_risk}<br/><b>Ice Thickness:</b> {ice_thickness} m",
            "style": {"backgroundColor": "#111827", "color": "white"}
        },
        map_style="light"
    )

    st.pydeck_chart(deck, use_container_width=True)
    st.caption("초록색은 낮은 BC Risk, 빨간색은 높은 BC Risk입니다. 파란 선은 AI가 위험지역을 피하도록 계산한 Dynamic Path입니다.")

with right:
    st.subheader("AI 운항 판정")

    risk_level = "LOW"
    if avg_path_risk > risk_grid["bc_risk"].quantile(0.70):
        risk_level = "HIGH"
    elif avg_path_risk > risk_grid["bc_risk"].quantile(0.45):
        risk_level = "MEDIUM"

    st.markdown(
        f"""
        <div class="risk-card">
        <h3>현재 항로 위험도: {risk_level}</h3>
        <p><b>평균 경로 BC Risk:</b> {avg_path_risk:,.1f}</p>
        <p><b>최대 경로 BC Risk:</b> {max_path_risk:,.1f}</p>
        <p><b>예상 CO₂eq:</b> {co2eq_tons:,.1f} tons</p>
        <p><b>연료소모 Index:</b> {fuel_consumption_index:,.1f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"AI 권장: 현재 {speed:.1f} knots에서 {recommended_speed:.1f} knots로 조정하면 "
        f"블랙카본 위험을 약 {bc_reduction:.1f}% 낮출 수 있습니다."
    )

    st.subheader("속도별 BC Risk 곡선")
    chart = alt.Chart(recommend_df).mark_line(point=True).encode(
        x=alt.X("speed:Q", title="Speed (knots)"),
        y=alt.Y("avg_bc_risk:Q", title="Average BC Risk Index"),
        tooltip=["speed", "avg_bc_risk", "days"]
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

st.divider()

# -----------------------------
# 9. Explainable model section
# -----------------------------
st.subheader("계산 로직")

formula_col1, formula_col2, formula_col3 = st.columns(3)
with formula_col1:
    st.markdown("""
    **1. 저항 계산**  
    `Resistance = 0.5 × speed² + 2.5 × ice_thickness × speed^1.5`
    """)
with formula_col2:
    st.markdown("""
    **2. BC Risk 계산**  
    `BC Risk = Resistance × Fuel Factor × Engine Load Multiplier`
    """)
with formula_col3:
    st.markdown("""
    **3. AI 경로 최적화**  
    `Cost = 0.35 × distance + 0.65 × BC Risk`
    """)

st.warning(
    "이 앱은 경진대회 발표용 프로토타입입니다. 현재 Heatmap은 합성 해빙 데이터로 생성되며, "
    "실제 분석에서는 NSIDC, Copernicus, AIS, IMO/ICCT 배출계수 데이터로 대체해야 합니다."
)

with st.expander("발표에서 강조할 해석 문장"):
    st.markdown(
        """
        - 북극항로의 환경성은 단순히 거리 단축으로 판단할 수 없다.
        - 빙하가 두꺼운 구간에서 속도를 높이면 엔진 부하와 블랙카본 위험이 비선형적으로 증가한다.
        - 따라서 미래 북극항로는 '가장 빠른 길'이 아니라 '가장 덜 녹이는 길'을 찾는 AI Routing이 필요하다.
        - Green Lane은 연료 전환뿐 아니라 실시간 속도 제어와 탄소 리스크 회피 능력을 갖춘 선박 중심으로 운영되어야 한다.
        """
    )
