import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import time

# page config
st.set_page_config(
    page_title="drone hub sim // accra",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom css
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  /* dark base */
  .stApp {
    background-color: #080c10;
    color: #e2ead4;
  }

  /* sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d1a;
  }
  [data-testid="stSidebar"] * {
    color: #b8d4a0 !important;
  }

  /* monospace for numbers */
  .mono { font-family: 'Share Tech Mono', monospace; }

  /* metric cards */
  .metric-card {
    background: #0d1a0a;
    border: 1px solid #2a4a20;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #5bff6a;
  }
  .metric-card.warn::before { background: #ffca3a; }
  .metric-card.bad::before  { background: #ff4d4d; }

  .metric-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5a7a50;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e2ead4;
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.7rem;
    color: #5a7a50;
    margin-top: 0.3rem;
    font-family: 'Share Tech Mono', monospace;
  }

  /* title block */
  .hero {
    border-bottom: 1px solid #1e2d1a;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }
  .hero-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #5bff6a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
  }
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #e2ead4;
    line-height: 1.1;
    margin: 0.3rem 0;
  }
  .hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: #4a6a40;
    font-size: 0.85rem;
  }

  /* section headers */
  .section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #5bff6a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1e2d1a;
    padding-bottom: 0.4rem;
  }

  /* run button */
  .stButton > button {
    background: #1a3a10 !important;
    color: #5bff6a !important;
    border: 1px solid #3a6a28 !important;
    border-radius: 3px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    width: 100%;
    padding: 0.6rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #2a5a18 !important;
    border-color: #5bff6a !important;
    color: #ffffff !important;
  }

  /* sliders */
  [data-testid="stSlider"] > div > div > div {
    background: #5bff6a !important;
  }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2d1a;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a6a40;
    border: none;
    padding: 0.5rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    color: #5bff6a !important;
    background: #0d1a0a !important;
    border-bottom: 2px solid #5bff6a !important;
  }

  /* spinner */
  .stSpinner > div {
    border-top-color: #5bff6a !important;
  }

  /* table */
  [data-testid="stDataFrame"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
  }

  /* expander */
  [data-testid="stExpander"] {
    border: 1px solid #1e2d1a;
    background: #0a130a;
    border-radius: 3px;
  }

  /* hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* progress bar */
  .stProgress > div > div {
    background: #5bff6a !important;
  }

  /* selectbox */
  [data-testid="stSelectbox"] select, .stSelectbox select {
    background: #0d1a0a;
    color: #e2ead4;
    border: 1px solid #2a4a20;
  }
</style>
""", unsafe_allow_html=True)

# constants
DRONE_SPEED       = 54.0
DRAIN_BASE        = 5.0
DRAIN_RETURN      = 3.57
HEAVY_MULT        = 1.5
RTH_THRESHOLD     = 25.0
RECHARGE_FULL_MIN = 26.5
BATTERY_EMERGENCY = 5.0
MAX_QUEUE_WAIT    = 90.0
HEAVY_PROB        = 0.30
DIST_MIN          = 3.0
DIST_MAX          = 8.0
LAMBDA_BY_HOUR    = {
    7: 2.0, 8: 2.0,
    9: 8.0, 10: 8.0, 11: 8.0,
    12: 5.0, 13: 5.0, 14: 5.0,
    15: 9.0, 16: 9.0, 17: 9.0, 18: 9.0,
    19: 3.0, 20: 3.0, 21: 3.0,
}
WIND_SEASONS = {
    'harmattan':  {'months': [11, 12, 1, 2], 'mult': 1.00, 'label': 'harmattan (nov–feb)'},
    'pre_rains':  {'months': [3, 4],         'mult': 1.15, 'label': 'pre-rains (mar–apr)'},
    'main_rains': {'months': [5, 6],         'mult': 1.25, 'label': 'main rains (may–jun)'},
    'mid_dry':    {'months': [7, 8, 9],      'mult': 1.25, 'label': 'mid-dry (jul–sep)'},
    'post_rains': {'months': [10],           'mult': 1.10, 'label': 'post-rains (oct)'},
}
PROCESS_NOISE_SD = 0.5
MEASURE_NOISE_SD = 0.8
SIM_START_HOUR   = 7
SIM_DURATION     = 900
SLA_LIMIT        = 120

# simulation logic
def get_wind_mult(month):
    for season in WIND_SEASONS.values():
        if month in season['months']:
            return season['mult']
    return 1.0

def get_lambda(sim_minutes):
    current_hour = SIM_START_HOUR + int(sim_minutes // 60)
    current_hour = min(current_hour, 21)
    return LAMBDA_BY_HOUR.get(current_hour, 3.0)

class Drone:
    def __init__(self, env, drone_id, chargers):
        self.env          = env
        self.id           = drone_id
        self.chargers     = chargers
        self.battery      = 100.0
        self.status       = 'idle'
        self.time_flying  = 0.0
        self.time_charging = 0.0
        self.time_waiting = 0.0

    def drain_battery(self, km, is_heavy, wind_mult, is_return=False):
        base   = DRAIN_RETURN if is_return else DRAIN_BASE
        mult   = HEAVY_MULT if is_heavy else 1.0
        noise  = np.random.normal(0, PROCESS_NOISE_SD * np.sqrt(km))
        drain  = (base * mult * wind_mult) * km + noise
        self.battery = max(0.0, self.battery - drain)

    def read_battery(self):
        noise = np.random.normal(0, MEASURE_NOISE_SD)
        return max(0.0, min(100.0, self.battery + noise))

    def charge(self):
        self.status    = 'waiting_for_charger'
        wait_start     = self.env.now
        with self.chargers.request() as req:
            yield req
            self.time_waiting += self.env.now - wait_start
            self.status        = 'charging'
            charge_start       = self.env.now
            deficit            = 100.0 - self.battery
            charge_time        = (deficit / 100.0) * RECHARGE_FULL_MIN
            yield self.env.timeout(charge_time)
            self.battery       = 100.0
            self.time_charging += self.env.now - charge_start
        self.status = 'idle'

def deliver_package(env, pkg_id, drone_pool, chargers, results, wind_mult, q_log):
    arrival  = env.now
    distance = np.random.uniform(DIST_MIN, DIST_MAX)
    is_heavy = np.random.random() < HEAVY_PROB
    fly_time = (distance / DRONE_SPEED) * 60

    q_log.append({'time': arrival, 'q_len': len(drone_pool.items)})

    drone     = yield drone_pool.get()
    drone.status = 'flying'
    wait_time = env.now - arrival

    yield env.timeout(fly_time)
    drone.drain_battery(distance, is_heavy, wind_mult, is_return=False)
    drone.time_flying += fly_time

    delivery_time = env.now - arrival
    results.append({
        'pkg_id':      pkg_id,
        'arrival_t':   arrival,
        'delivery_t':  delivery_time,
        'wait_t':      wait_time,
        'flight_t':    fly_time,
        'distance_km': round(distance, 2),
        'is_heavy':    is_heavy,
        'late':        delivery_time > SLA_LIMIT,
        'drone_id':    drone.id,
        'wind_mult':   wind_mult,
    })

    yield env.timeout(fly_time)
    drone.drain_battery(distance, False, wind_mult, is_return=True)
    drone.time_flying += fly_time

    if drone.read_battery() < RTH_THRESHOLD:
        yield env.process(drone.charge())

    drone.status = 'idle'
    yield drone_pool.put(drone)

def package_generator(env, drone_pool, chargers, results, wind_mult, q_log):
    pkg_id = 0
    while True:
        lam          = get_lambda(env.now)
        interarrival = np.random.exponential(60.0 / lam)
        yield env.timeout(interarrival)
        env.process(deliver_package(env, pkg_id, drone_pool, chargers, results, wind_mult, q_log))
        pkg_id += 1

def run_simulation(n_drones, n_chargers, seed, month=1):
    np.random.seed(seed)
    env        = simpy.Environment()
    chargers   = simpy.Resource(env, capacity=n_chargers)
    drone_pool = simpy.FilterStore(env)
    results    = []
    q_log      = []
    wind_mult  = get_wind_mult(month)
    drones     = [Drone(env, i, chargers) for i in range(n_drones)]
    for drone in drones:
        drone_pool.put(drone)
    env.process(package_generator(env, drone_pool, chargers, results, wind_mult, q_log))
    env.run(until=SIM_DURATION)
    return results, q_log, drones

def extract_metrics(results, drones, n_drones, n_chargers, seed, month):
    df = pd.DataFrame(results)
    if len(df) == 0:
        return None
    total_time   = n_drones * SIM_DURATION
    total_flying = sum(d.time_flying   for d in drones)
    total_charge = sum(d.time_charging for d in drones)
    total_wait   = sum(d.time_waiting  for d in drones)
    return {
        'n_drones':        n_drones,
        'n_chargers':      n_chargers,
        'seed':            seed,
        'month':           month,
        'total_packages':  len(df),
        'late_count':      int(df['late'].sum()),
        'on_time_rate':    (~df['late']).mean(),
        'mean_delivery_t': df['delivery_t'].mean(),
        'p95_delivery_t':  df['delivery_t'].quantile(0.95),
        'p50_delivery_t':  df['delivery_t'].median(),
        'mean_wait_t':     df['wait_t'].mean(),
        'p95_wait_t':      df['wait_t'].quantile(0.95),
        'drone_util_fly':  total_flying / total_time,
        'drone_util_charge': total_charge / total_time,
        'drone_util_wait': total_wait / total_time,
    }

# plotly theme
PLOT_BG    = '#080c10'
PLOT_PAPER = '#080c10'
GRID_COL   = '#1e2d1a'
TEXT_COL   = '#b8d4a0'
GREEN      = '#5bff6a'
RED        = '#ff4d4d'
YELLOW     = '#ffca3a'
BLUE       = '#4ab8ff'
PALETTE    = [GREEN, BLUE, YELLOW, RED, '#c97bff', '#ff8c42']

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family='Share Tech Mono, monospace', color=TEXT_COL, size=11),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID_COL),
    )
    return fig

# sidebar
with st.sidebar:
    st.markdown('<div class="section-label">// fleet config</div>', unsafe_allow_html=True)
    n_drones   = st.slider("drones", 3, 10, 6, 1)
    n_chargers = st.slider("charging stations", 1, 6, 2, 1)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">// environment</div>', unsafe_allow_html=True)
    season_names = list(WIND_SEASONS.keys())
    season_labels_map = {k: v['label'] for k, v in WIND_SEASONS.items()}
    selected_season = st.selectbox(
        "ghanaian season",
        options=season_names,
        format_func=lambda x: season_labels_map[x],
        index=0,
    )
    month = WIND_SEASONS[selected_season]['months'][0]

    st.markdown('<div class="section-label" style="margin-top:1.5rem">// experiment</div>', unsafe_allow_html=True)
    n_reps = st.slider("replications", 5, 30, 10, 5)
    seed   = st.number_input("random seed", value=42, min_value=0, step=1)

    st.markdown('<div class="section-label" style="margin-top:1.5rem">// sla</div>', unsafe_allow_html=True)
    sla_target = st.slider("target on-time rate (%)", 80, 99, 95, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    run_single = st.button("▶  run single simulation")
    run_grid   = st.button("⬛  run full grid search")

    wind_mult = WIND_SEASONS[selected_season]['mult']
    st.markdown(f"""
    <div style="margin-top:1.5rem; padding:0.8rem; background:#0a130a; border:1px solid #1e2d1a; border-radius:3px;">
        <div class="section-label" style="margin-bottom:0.5rem">// active params</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#4a6a40; line-height:1.8;">
        wind mult: <span style="color:#b8d4a0">{wind_mult:.2f}x</span><br>
        dist range: <span style="color:#b8d4a0">3–8 km</span><br>
        sla window: <span style="color:#b8d4a0">120 min</span><br>
        sim duration: <span style="color:#b8d4a0">15 hrs</span><br>
        heavy pkg prob: <span style="color:#b8d4a0">30%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# hero header
st.markdown("""
<div class="hero">
  <div class="hero-tag">// last-mile drone delivery · accra, ghana · discrete-event simulation</div>
  <div class="hero-title">drone hub<br>control room</div>
  <div class="hero-sub">simpy-powered fleet optimizer · DJI FlyCart 30 specs · accra wind seasons</div>
</div>
""", unsafe_allow_html=True)

# session state
if 'single_results' not in st.session_state:
    st.session_state.single_results = None
if 'grid_results' not in st.session_state:
    st.session_state.grid_results = None

# single sim run
if run_single:
    with st.spinner("running simulation..."):
        results, q_log, drones = run_simulation(n_drones, n_chargers, seed, month)
    st.session_state.single_results = (results, q_log, drones, n_drones, n_chargers, month)

# grid search run
if run_grid:
    drone_range   = list(range(3, 11))
    charger_range = list(range(1, 6))
    total_runs    = len(drone_range) * len(charger_range) * n_reps
    rows          = []

    pb    = st.progress(0)
    done  = 0

    for nd in drone_range:
        for nc in charger_range:
            for s in range(1, n_reps + 1):
                r, _, dr = run_simulation(nd, nc, s, month)
                m = extract_metrics(r, dr, nd, nc, s, month)
                if m:
                    rows.append(m)
                done += 1
                pb.progress(done / total_runs)

    raw_df   = pd.DataFrame(rows)
    summary  = (
        raw_df.groupby(['n_drones', 'n_chargers'])
        .agg(
            on_time_mean  = ('on_time_rate', 'mean'),
            on_time_ci    = ('on_time_rate', lambda x: stats.sem(x) * stats.t.ppf(0.975, df=len(x)-1)),
            p95_mean      = ('p95_delivery_t', 'mean'),
            late_mean     = ('late_count', 'mean'),
        )
        .reset_index()
    )
    summary['hits_sla']   = summary['on_time_mean'] >= (sla_target / 100)
    summary['cost_proxy'] = summary['n_drones'] * 3 + summary['n_chargers']

    st.session_state.grid_results = (raw_df, summary)
    pb.empty()

# display single results
if st.session_state.single_results:
    results, q_log, drones, nd, nc, mo = st.session_state.single_results
    df = pd.DataFrame(results)
    metrics = extract_metrics(results, drones, nd, nc, seed, mo)

    on_time  = metrics['on_time_rate']
    p95      = metrics['p95_delivery_t']
    late_pct = metrics['late_count'] / metrics['total_packages']

    st.markdown('<div class="section-label">// single run results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_card(col, label, value, sub, cls=''):
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    metric_card(c1, "on-time rate", f"{on_time:.1%}",
                f"sla target: {sla_target}%",
                '' if on_time >= sla_target/100 else 'bad')
    metric_card(c2, "total packages", f"{metrics['total_packages']}",
                f"late: {metrics['late_count']}")
    metric_card(c3, "p95 delivery", f"{p95:.0f}m",
                f"mean: {metrics['mean_delivery_t']:.0f}m",
                '' if p95 <= SLA_LIMIT else 'warn')
    metric_card(c4, "mean wait time", f"{metrics['mean_wait_t']:.1f}m",
                f"p95: {metrics['p95_wait_t']:.1f}m")
    metric_card(c5, "fleet utilisation", f"{metrics['drone_util_fly']*100:.0f}%",
                f"charging: {metrics['drone_util_charge']*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["delivery times", "queue dynamics", "drone utilisation", "raw data"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">delivery time histogram</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['delivery_t'],
                nbinsx=40,
                marker_color=GREEN,
                opacity=0.7,
                name='delivery time',
            ))
            fig.add_vline(x=SLA_LIMIT, line_dash='dash', line_color=RED, line_width=2,
                          annotation_text='120m sla', annotation_font_color=RED)
            fig.update_layout(title='delivery time distribution', xaxis_title='minutes', yaxis_title='count')
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-label">cdf of delivery times</div>', unsafe_allow_html=True)
            sorted_t = np.sort(df['delivery_t'].values)
            cdf      = np.arange(1, len(sorted_t)+1) / len(sorted_t)
            fig2     = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sorted_t, y=cdf * 100,
                mode='lines', line=dict(color=GREEN, width=2.5),
                name='CDF',
            ))
            fig2.add_vline(x=SLA_LIMIT, line_dash='dash', line_color=RED, line_width=2)
            fig2.add_hline(y=sla_target, line_dash='dot', line_color=YELLOW, line_width=1.5)
            fig2.update_layout(title='cumulative distribution', xaxis_title='delivery time (min)',
                               yaxis_title='cumulative %')
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-label">package arrival demand by hour</div>', unsafe_allow_html=True)
        hours  = list(range(7, 22))
        lambda_vals = [LAMBDA_BY_HOUR.get(h, 0) for h in hours]
        fig3   = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=lambda_vals,
            marker_color=[GREEN if l >= 8 else YELLOW if l >= 5 else BLUE for l in lambda_vals],
            name='λ (pkg/hr)',
        ))
        fig3.update_layout(title='time-varying arrival rate λ', xaxis_title='hour of day',
                           yaxis_title='packages per hour')
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        if q_log:
            st.markdown('<div class="section-label">queue depth over simulation time</div>', unsafe_allow_html=True)
            qdf   = pd.DataFrame(q_log)
            fig4  = go.Figure()
            fig4.add_trace(go.Scatter(
                x=qdf['time'], y=qdf['q_len'],
                mode='lines', fill='tozeroy',
                line=dict(color=BLUE, width=1.5),
                fillcolor='rgba(74,184,255,0.1)',
                name='available drones',
            ))
            fig4.update_layout(title='available drones in pool over time',
                               xaxis_title='sim time (min)', yaxis_title='drones available')
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-label">individual drone time breakdown</div>', unsafe_allow_html=True)
        util_data = []
        for d in drones:
            total_acc = d.time_flying + d.time_charging + d.time_waiting
            idle_t    = max(0, SIM_DURATION - total_acc)
            util_data.append({
                'drone':    f'drone {d.id:02d}',
                'flying':   round(d.time_flying / SIM_DURATION * 100, 1),
                'charging': round(d.time_charging / SIM_DURATION * 100, 1),
                'waiting':  round(d.time_waiting / SIM_DURATION * 100, 1),
                'idle':     round(idle_t / SIM_DURATION * 100, 1),
            })
        util_df = pd.DataFrame(util_data)

        fig5 = go.Figure()
        for col, color in [('flying', GREEN), ('charging', BLUE), ('waiting', RED), ('idle', '#3a4a38')]:
            fig5.add_trace(go.Bar(
                name=col, x=util_df['drone'], y=util_df[col],
                marker_color=color, opacity=0.85,
            ))
        fig5.update_layout(barmode='stack', title='drone time utilisation (%)',
                           yaxis_title='% of sim duration', xaxis_title='')
        apply_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-label">delivery log</div>', unsafe_allow_html=True)
        display_df = df[['pkg_id','arrival_t','delivery_t','wait_t','distance_km','is_heavy','late','drone_id']].copy()
        display_df['arrival_t']   = display_df['arrival_t'].round(1)
        display_df['delivery_t']  = display_df['delivery_t'].round(1)
        display_df['wait_t']      = display_df['wait_t'].round(1)
        st.dataframe(display_df, use_container_width=True, height=350)

# display grid results
if st.session_state.grid_results:
    raw_df, summary = st.session_state.grid_results

    sla_configs  = summary[summary['hits_sla']].copy()
    optimal      = sla_configs.loc[sla_configs['cost_proxy'].idxmin()] if len(sla_configs) > 0 else None

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">// grid search results</div>', unsafe_allow_html=True)

    if optimal is not None:
        opt_d = int(optimal['n_drones'])
        opt_c = int(optimal['n_chargers'])
        oc1, oc2, oc3 = st.columns(3)
        metric_card(oc1, "optimal config", f"{opt_d}D + {opt_c}C", f"cost proxy: {int(optimal['cost_proxy'])}")
        metric_card(oc2, "on-time rate",
                    f"{optimal['on_time_mean']:.1%}",
                    f"±{optimal['on_time_ci']:.1%} CI")
        metric_card(oc3, "p95 delivery", f"{optimal['p95_mean']:.0f}m", f"late/run: {optimal['late_mean']:.1f}")
    else:
        st.warning("no configuration met the sla target. try increasing drones/chargers.")
        opt_d, opt_c = None, None

    st.markdown("<br>", unsafe_allow_html=True)

    gtab1, gtab2, gtab3 = st.tabs(["heatmap", "fleet size vs on-time", "utilisation breakdown"])

    with gtab1:
        st.markdown('<div class="section-label">on-time rate heatmap by fleet config</div>', unsafe_allow_html=True)
        heat_pivot = summary.pivot(index='n_drones', columns='n_chargers', values='on_time_mean') * 100

        fig_h = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=[f"{c}C" for c in heat_pivot.columns],
            y=[f"{d}D" for d in heat_pivot.index],
            colorscale=[[0, '#300a0a'], [0.5, '#3a5a10'], [1, '#5bff6a']],
            zmin=50, zmax=100,
            text=np.round(heat_pivot.values, 1),
            texttemplate='%{text}%',
            textfont=dict(family='Share Tech Mono', size=11),
            colorbar=dict(title='on-time %', tickfont=dict(color=TEXT_COL)),
        ))
        if opt_d and opt_c:
            opt_x = list(heat_pivot.columns).index(opt_c)
            opt_y = list(heat_pivot.index).index(opt_d)
            fig_h.add_shape(type='rect',
                x0=opt_x-0.5, x1=opt_x+0.5,
                y0=opt_y-0.5, y1=opt_y+0.5,
                line=dict(color=YELLOW, width=3, dash='dash'),
            )
            fig_h.add_annotation(
                x=opt_x, y=opt_y - 0.6,
                text='optimal', font=dict(color=YELLOW, size=9),
                showarrow=False,
            )
        fig_h.update_layout(title='on-time delivery rate (%) — black = passes sla, yellow dashed = optimal',
                            xaxis_title='charging stations', yaxis_title='drones')
        apply_theme(fig_h)
        st.plotly_chart(fig_h, use_container_width=True)

    with gtab2:
        st.markdown('<div class="section-label">on-time rate vs fleet size</div>', unsafe_allow_html=True)
        fig_l = go.Figure()
        for idx, nc in enumerate(sorted(summary['n_chargers'].unique())):
            subset = summary[summary['n_chargers'] == nc].sort_values('n_drones')
            color  = PALETTE[idx % len(PALETTE)]
            fig_l.add_trace(go.Scatter(
                x=subset['n_drones'],
                y=subset['on_time_mean'] * 100,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=7),
                name=f'{int(nc)} chargers',
                error_y=dict(
                    type='data',
                    array=(subset['on_time_ci'] * 100).values,
                    visible=True,
                    color=color,
                    thickness=1,
                ),
            ))
        fig_l.add_hline(y=sla_target, line_dash='dash', line_color='white', line_width=1.5,
                        annotation_text=f'{sla_target}% sla', annotation_font_color='white')
        fig_l.update_layout(title='on-time rate vs fleet size', xaxis_title='number of drones',
                            yaxis_title='on-time rate (%)', yaxis_range=[40, 105])
        apply_theme(fig_l)
        st.plotly_chart(fig_l, use_container_width=True)

    with gtab3:
        if opt_c:
            st.markdown('<div class="section-label">drone time breakdown at optimal charger count</div>', unsafe_allow_html=True)
            util_s = summary[summary['n_chargers'] == opt_c].sort_values('n_drones')

            fig_u = go.Figure()
            for col, color, label in [
                ('drone_util_fly', GREEN, 'flying'),
                ('drone_util_charge', BLUE, 'charging'),
                ('drone_util_wait', RED, 'waiting for charger'),
            ]:
                vals = raw_df[raw_df['n_chargers'] == opt_c].groupby('n_drones')[col].mean() * 100 if col in raw_df.columns else util_s[col] * 100
                nd_vals = sorted(raw_df['n_drones'].unique()) if col in raw_df.columns else util_s['n_drones'].tolist()
                fig_u.add_trace(go.Bar(
                    x=[f"{int(d)}D" for d in util_s['n_drones']],
                    y=util_s[col] * 100,
                    name=label,
                    marker_color=color,
                    opacity=0.85,
                ))
            fig_u.update_layout(barmode='stack', title=f'utilisation at {opt_c} chargers',
                                xaxis_title='fleet size', yaxis_title='% of total drone time')
            apply_theme(fig_u)
            st.plotly_chart(fig_u, use_container_width=True)

# noise verification (always shown)
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("// noise model verification", expanded=False):
    st.markdown('<div class="section-label">battery drain & sensor noise distributions</div>', unsafe_allow_html=True)

    np.random.seed(0)
    dist_km  = 5.0
    n_trials = 1000
    drain_samples = [
        (DRAIN_BASE * 1.0 * 1.0) * dist_km + np.random.normal(0, PROCESS_NOISE_SD * np.sqrt(dist_km))
        for _ in range(n_trials)
    ]
    sensor_samples = [
        26.0 + np.random.normal(0, MEASURE_NOISE_SD)
        for _ in range(n_trials)
    ]

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        fig_n1 = go.Figure()
        fig_n1.add_trace(go.Histogram(x=drain_samples, nbinsx=40, marker_color=GREEN, opacity=0.7))
        fig_n1.add_vline(x=np.mean(drain_samples), line_dash='dash', line_color=YELLOW)
        fig_n1.update_layout(title=f'battery drain noise — {dist_km}km trip (n=1000)',
                             xaxis_title='% drained', yaxis_title='count')
        apply_theme(fig_n1)
        st.plotly_chart(fig_n1, use_container_width=True)

    with col_n2:
        fig_n2 = go.Figure()
        fig_n2.add_trace(go.Histogram(x=sensor_samples, nbinsx=40, marker_color=BLUE, opacity=0.7))
        fig_n2.add_vline(x=26.0, line_dash='dash', line_color=YELLOW,
                         annotation_text='true = 26%', annotation_font_color=YELLOW)
        fig_n2.update_layout(title='battery sensor noise — true=26% (n=1000)',
                             xaxis_title='reported %', yaxis_title='count')
        apply_theme(fig_n2)
        st.plotly_chart(fig_n2, use_container_width=True)

# footer
st.markdown("""
<div style="margin-top:3rem; border-top:1px solid #1e2d1a; padding-top:1rem;
     font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:#2a4a20;
     display:flex; justify-content:space-between;">
  <span>group 2 · last-mile drone delivery hub simulation</span>
  <span>accra, ghana · DJI FlyCart 30 · simpy v4</span>
</div>
""", unsafe_allow_html=True)