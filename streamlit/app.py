import streamlit as st
import streamlit.components.v1 as components
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import json

st.set_page_config(
    page_title="drone hub sim — accra",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

if dark:
    BG        = "#191919"
    BG2       = "#202020"
    BG3       = "#2f2f2f"
    BORDER    = "#373737"
    TEXT      = "#e6e6e6"
    TEXT2     = "#9b9b9b"
    TEXT3     = "#686868"
    ACCENT    = "#2383e2"
    RED_C     = "#eb4d3d"
    GREEN_C   = "#3ea85d"
    YELLOW_C  = "#dfb862"
    PLOT_BG   = "#202020"
    PLOT_GRID = "#2f2f2f"
    PLOT_TEXT = "#9b9b9b"
else:
    BG        = "#ffffff"
    BG2       = "#f7f6f3"
    BG3       = "#efefef"
    BORDER    = "#e9e9e7"
    TEXT      = "#37352f"
    TEXT2     = "#787774"
    TEXT3     = "#acaba8"
    ACCENT    = "#2383e2"
    RED_C     = "#eb4d3d"
    GREEN_C   = "#3ea85d"
    YELLOW_C  = "#c88b22"
    PLOT_BG   = "#ffffff"
    PLOT_GRID = "#f0f0f0"
    PLOT_TEXT = "#787774"

PALETTE = [ACCENT, GREEN_C, YELLOW_C, RED_C, "#9065b0", "#e06c2d"]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT};
  }}
  [data-testid="stSidebar"] {{
    background-color: {BG2} !important;
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
  [data-testid="stSidebar"] label {{
    color: {TEXT2} !important;
    font-size: 0.8rem !important;
  }}
  .block-container {{ padding: 2rem 2.5rem 4rem; max-width: 1400px; }}
  h1, h2, h3 {{ font-weight: 600; color: {TEXT}; }}

  .notion-title {{
    font-size: 2.4rem; font-weight: 700; color: {TEXT};
    letter-spacing: -0.02em; margin-bottom: 0.2rem; line-height: 1.2;
  }}
  .notion-sub {{ font-size: 0.88rem; color: {TEXT2}; margin-bottom: 1.5rem; }}
  .notion-divider {{ border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0; }}
  .notion-section {{
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: {TEXT3}; margin-bottom: 0.6rem; margin-top: 1.2rem;
  }}

  .kpi-card {{
    background: {BG2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 1rem 1.2rem;
  }}
  .kpi-label {{ font-size: 0.7rem; color: {TEXT2}; margin-bottom: 0.25rem; font-weight: 500; }}
  .kpi-value {{ font-size: 1.7rem; font-weight: 600; color: {TEXT}; line-height: 1.1; }}
  .kpi-sub   {{ font-size: 0.7rem; color: {TEXT3}; margin-top: 0.15rem; }}
  .kpi-good .kpi-value {{ color: {GREEN_C}; }}
  .kpi-warn .kpi-value {{ color: {YELLOW_C}; }}
  .kpi-bad  .kpi-value {{ color: {RED_C}; }}

  .stButton > button {{
    background: {ACCENT} !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 0.5rem 1rem !important;
    width: 100%; transition: opacity 0.15s !important;
  }}
  .stButton > button:hover {{ opacity: 0.82 !important; }}

  .stTabs [data-baseweb="tab-list"] {{
    background: transparent; border-bottom: 1px solid {BORDER}; gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif; font-size: 0.82rem; font-weight: 500;
    color: {TEXT2}; border: none; padding: 0.6rem 1rem; background: transparent;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important; background: transparent !important;
    border-bottom: 2px solid {TEXT} !important;
  }}
  [data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important; border-radius: 8px !important;
    background: {BG2} !important;
  }}
  .stProgress > div > div {{ background: {ACCENT} !important; }}
  .stSpinner > div {{ border-top-color: {ACCENT} !important; }}
  #MainMenu, footer {{ visibility: hidden; }}
  header {
    visibility: visible;
    height: 0px;
  }

  .tag {{
    display: inline-block; background: {BG3}; color: {TEXT2};
    border-radius: 4px; font-size: 0.7rem; font-weight: 500;
    padding: 0.12rem 0.45rem; margin-right: 0.25rem;
  }}
  .tag-blue  {{ background: {'#1e3a5f' if dark else '#e8f1fb'}; color: {ACCENT}; }}
  .tag-green {{ background: {'#1a3327' if dark else '#e9f5ee'}; color: {GREEN_C}; }}
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
        self.env           = env
        self.id            = drone_id
        self.chargers      = chargers
        self.battery       = 100.0
        self.status        = 'idle'
        self.time_flying   = 0.0
        self.time_charging = 0.0
        self.time_waiting  = 0.0

    def drain_battery(self, km, is_heavy, wind_mult, is_return=False):
        base  = DRAIN_RETURN if is_return else DRAIN_BASE
        mult  = HEAVY_MULT if is_heavy else 1.0
        noise = np.random.normal(0, PROCESS_NOISE_SD * np.sqrt(km))
        drain = (base * mult * wind_mult) * km + noise
        self.battery = max(0.0, self.battery - drain)

    def read_battery(self):
        noise = np.random.normal(0, MEASURE_NOISE_SD)
        return max(0.0, min(100.0, self.battery + noise))

    def charge(self):
        self.status = 'waiting_for_charger'
        wait_start  = self.env.now
        with self.chargers.request() as req:
            yield req
            self.time_waiting  += self.env.now - wait_start
            self.status         = 'charging'
            charge_start        = self.env.now
            deficit             = 100.0 - self.battery
            charge_time         = (deficit / 100.0) * RECHARGE_FULL_MIN
            yield self.env.timeout(charge_time)
            self.battery        = 100.0
            self.time_charging += self.env.now - charge_start
        self.status = 'idle'

def deliver_package(env, pkg_id, drone_pool, chargers, results, wind_mult, q_log, flight_log):
    arrival  = env.now
    distance = np.random.uniform(DIST_MIN, DIST_MAX)
    is_heavy = np.random.random() < HEAVY_PROB
    fly_time = (distance / DRONE_SPEED) * 60

    q_log.append({'time': arrival, 'q_len': len(drone_pool.items)})

    drone     = yield drone_pool.get()
    drone.status = 'flying'
    wait_time = env.now - arrival
    depart_t  = env.now

    flight_log.append({
        'drone_id':  drone.id,
        'pkg_id':    pkg_id,
        'depart_t':  depart_t,
        'arrive_t':  depart_t + fly_time,
        'return_t':  depart_t + fly_time * 2,
        'distance':  round(distance, 2),
        'is_heavy':  is_heavy,
    })

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

def package_generator(env, drone_pool, chargers, results, wind_mult, q_log, flight_log):
    pkg_id = 0
    while True:
        lam          = get_lambda(env.now)
        interarrival = np.random.exponential(60.0 / lam)
        yield env.timeout(interarrival)
        env.process(deliver_package(env, pkg_id, drone_pool, chargers, results, wind_mult, q_log, flight_log))
        pkg_id += 1

def run_simulation(n_drones, n_chargers, seed, month=1):
    np.random.seed(seed)
    env        = simpy.Environment()
    chargers   = simpy.Resource(env, capacity=n_chargers)
    drone_pool = simpy.FilterStore(env)
    results    = []
    q_log      = []
    flight_log = []
    wind_mult  = get_wind_mult(month)
    drones     = [Drone(env, i, chargers) for i in range(n_drones)]
    for drone in drones:
        drone_pool.put(drone)
    env.process(package_generator(env, drone_pool, chargers, results, wind_mult, q_log, flight_log))
    env.run(until=SIM_DURATION)
    return results, q_log, drones, flight_log

def extract_metrics(results, drones, n_drones, n_chargers, seed, month):
    df = pd.DataFrame(results)
    if len(df) == 0:
        return None
    total_time   = n_drones * SIM_DURATION
    total_flying = sum(d.time_flying   for d in drones)
    total_charge = sum(d.time_charging for d in drones)
    total_wait   = sum(d.time_waiting  for d in drones)
    return {
        'n_drones':          n_drones,
        'n_chargers':        n_chargers,
        'seed':              seed,
        'month':             month,
        'total_packages':    len(df),
        'late_count':        int(df['late'].sum()),
        'on_time_rate':      (~df['late']).mean(),
        'mean_delivery_t':   df['delivery_t'].mean(),
        'p95_delivery_t':    df['delivery_t'].quantile(0.95),
        'p50_delivery_t':    df['delivery_t'].median(),
        'mean_wait_t':       df['wait_t'].mean(),
        'p95_wait_t':        df['wait_t'].quantile(0.95),
        'drone_util_fly':    total_flying  / total_time,
        'drone_util_charge': total_charge  / total_time,
        'drone_util_wait':   total_wait    / total_time,
    }

def apply_theme(fig, height=380):
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family='Inter, sans-serif', color=PLOT_TEXT, size=11),
        xaxis=dict(gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zerolinecolor=PLOT_GRID),
        yaxis=dict(gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zerolinecolor=PLOT_GRID),
        margin=dict(l=40, r=20, t=36, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
        height=height,
    )
    return fig

def build_flight_animation(flight_log, n_drones, dark_mode):
    sample     = flight_log[:150]
    bg         = "#191919" if dark_mode else "#ffffff"
    bg2        = "#2f2f2f" if dark_mode else "#f7f6f3"
    border_c   = "#373737" if dark_mode else "#e9e9e7"
    text_c     = "#e6e6e6" if dark_mode else "#37352f"
    text2_c    = "#9b9b9b" if dark_mode else "#787774"
    bg3_c      = "#3a3a3a" if dark_mode else "#efefef"

    drone_colors = ["#2383e2","#3ea85d","#eb4d3d","#dfb862","#9065b0","#e06c2d","#3ea8a8","#e06c8c"]
    flights_json = json.dumps(sample)
    n_drones_js  = n_drones

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:{bg};font-family:'Inter',sans-serif;overflow:hidden}}
  canvas{{display:block}}
  #ui{{position:absolute;top:12px;left:12px;right:12px;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none}}
  #legend{{background:{bg2};border:1px solid {border_c};border-radius:8px;padding:8px 12px;font-size:11px;color:{text2_c}}}
  #legend div{{margin:3px 0;display:flex;align-items:center;gap:6px}}
  .dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
  #controls{{background:{bg2};border:1px solid {border_c};border-radius:8px;padding:8px 12px;pointer-events:all}}
  #controls label{{font-size:11px;color:{text2_c};display:block;margin-bottom:4px}}
  #speedSlider{{width:110px;accent-color:#2383e2}}
  #timeLabel{{font-size:11px;color:{text_c};font-weight:600;margin-top:4px}}
  #playBtn{{background:#2383e2;color:white;border:none;border-radius:5px;font-size:11px;font-weight:500;padding:4px 10px;cursor:pointer;margin-top:6px;width:100%}}
  #statusBar{{position:absolute;bottom:12px;left:12px;right:12px;display:flex;gap:8px;flex-wrap:wrap}}
  .chip{{background:{bg2};border:1px solid {border_c};border-radius:6px;padding:4px 10px;font-size:11px;color:{text2_c}}}
  .chip span{{color:{text_c};font-weight:600}}

  /* Fix dropdowns / selectboxes */
  [data-baseweb="select"] > div {
  background-color: ${BG2} !important;
  border-color: ${BORDER} !important;
  color: ${TEXT} !important;}

    /* Dropdown menu */
    div[role="listbox"] {
      background-color: ${BG} !important;
      border: 1px solid ${BORDER} !important;
    }

    /* Dropdown options */
    div[role="option"] {
      background-color: ${BG} !important;
      color: ${TEXT} !important;
    }

    /* Hover state */
    div[role="option"]:hover {
      background-color: ${BG3} !important;
    }

    /* Selected option */
    div[aria-selected="true"] {
      background-color: ${BG3} !important;
      color: ${TEXT} !important;
    }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="ui">
  <div id="legend">
    <div><div class="dot" style="background:#2383e2"></div>outbound</div>
    <div><div class="dot" style="background:#3ea85d"></div>return</div>
    <div><div class="dot" style="background:#dfb862"></div>charging</div>
    <div><div class="dot" style="background:{bg3_c}"></div>idle</div>
  </div>
  <div id="controls">
    <label>speed: <strong id="speedVal">8×</strong></label>
    <input id="speedSlider" type="range" min="1" max="40" value="8">
    <div id="timeLabel">t = 0 min</div>
    <button id="playBtn">⏸ pause</button>
  </div>
</div>
<div id="statusBar">
  <div class="chip">drones: <span>{n_drones_js}</span></div>
  <div class="chip">flights: <span>{len(sample)}</span></div>
  <div class="chip">active: <span id="activeCount">0</span></div>
  <div class="chip">delivered: <span id="delivCount">0</span></div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
function W(){{ return window.innerWidth; }}
function H(){{ return window.innerHeight; }}
function resize(){{ canvas.width=W(); canvas.height=H(); }}
resize();
window.addEventListener('resize', resize);

const flights  = {flights_json};
const N_DRONES = {n_drones_js};
const COLORS   = {json.dumps(drone_colors)};
const SIM_START = {SIM_START_HOUR};

function HX(){{ return W()*0.5; }}
function HY(){{ return H()*0.5; }}
function RAD(){{ return Math.min(W(),H())*0.33; }}

const destAngles = {{}};
flights.forEach((f,i)=>{{
  if(!(f.pkg_id in destAngles))
    destAngles[f.pkg_id] = (Math.PI*2*i/flights.length) - Math.PI/2;
}});

let simTime = 0, speed = 8, running = true, lastTs = null;

document.getElementById('speedSlider').addEventListener('input', e=>{{
  speed=+e.target.value;
  document.getElementById('speedVal').textContent=speed+'×';
}});
document.getElementById('playBtn').addEventListener('click', ()=>{{
  running=!running;
  document.getElementById('playBtn').textContent=running?'⏸ pause':'▶ play';
}});

const droneState = Array.from({{length:N_DRONES}},(_,i)=>({{
  id:i, x:HX(), y:HY(), status:'idle', battery:100, trail:[]
}}));

function ease(t){{ return t<0.5?2*t*t:-1+(4-2*t)*t; }}
function destXY(angle){{
  return [HX()+Math.cos(angle)*RAD(), HY()+Math.sin(angle)*RAD()];
}}

function drawHub(){{
  const x=HX(), y=HY();
  const g=ctx.createRadialGradient(x,y,0,x,y,45);
  g.addColorStop(0,'#2383e244'); g.addColorStop(1,'transparent');
  ctx.beginPath(); ctx.arc(x,y,45,0,Math.PI*2);
  ctx.fillStyle=g; ctx.fill();
  [30,20,12].forEach((r,i)=>{{
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
    ctx.strokeStyle='#2383e2'+['44','88','cc'][i];
    ctx.lineWidth=1; ctx.stroke();
  }});
  ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
  ctx.fillStyle='#2383e2'; ctx.fill();
  ctx.font='500 11px Inter,sans-serif';
  ctx.fillStyle='{text2_c}';
  ctx.textAlign='center';
  ctx.fillText('HUB',x,y+44);
}}

function drawDest(angle, active){{
  const [x,y]=destXY(angle);
  ctx.beginPath(); ctx.arc(x,y,active?8:4,0,Math.PI*2);
  ctx.fillStyle=active?'#3ea85d33':'{border_c}';
  ctx.fill();
  ctx.beginPath(); ctx.arc(x,y,active?4.5:2.5,0,Math.PI*2);
  ctx.fillStyle=active?'#3ea85d':'{bg3_c}'; ctx.fill();
}}

function drawFlightPath(angle, color){{
  const [x,y]=destXY(angle);
  ctx.setLineDash([4,7]);
  ctx.beginPath(); ctx.moveTo(HX(),HY()); ctx.lineTo(x,y);
  ctx.strokeStyle=color+'2a'; ctx.lineWidth=1; ctx.stroke();
  ctx.setLineDash([]);
}}

function drawDrone(d){{
  const color=COLORS[d.id%COLORS.length];
  if(d.trail.length>1){{
    ctx.beginPath(); ctx.moveTo(d.trail[0].x,d.trail[0].y);
    d.trail.forEach(p=>ctx.lineTo(p.x,p.y));
    ctx.strokeStyle=color+'55'; ctx.lineWidth=1.5; ctx.stroke();
  }}
  if(d.status==='idle') return;
  ctx.save(); ctx.translate(d.x,d.y);
  ctx.beginPath(); ctx.arc(0,0,11,0,Math.PI*2);
  ctx.fillStyle=color+'1a'; ctx.fill();
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sy])=>{{
    ctx.beginPath(); ctx.moveTo(sx*2,sy*2); ctx.lineTo(sx*7,sy*7);
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.lineCap='round'; ctx.stroke();
    ctx.beginPath(); ctx.arc(sx*7,sy*7,3,0,Math.PI*2);
    ctx.strokeStyle=color+'99'; ctx.lineWidth=1.2; ctx.stroke();
  }});
  ctx.beginPath(); ctx.arc(0,0,2.5,0,Math.PI*2);
  ctx.fillStyle=color; ctx.fill();
  ctx.restore();
  ctx.font='500 9px Inter,sans-serif';
  ctx.fillStyle=color; ctx.textAlign='center';
  ctx.fillText('D'+d.id, d.x, d.y-15);
  const bw=22, bh=3, bx=d.x-11, by=d.y+15;
  ctx.fillStyle='{bg3_c}'; ctx.fillRect(bx,by,bw,bh);
  const pct=Math.max(0,Math.min(1,d.battery/100));
  ctx.fillStyle=pct>0.5?'#3ea85d':pct>0.25?'#dfb862':'#eb4d3d';
  ctx.fillRect(bx,by,bw*pct,bh);
}}

function frame(ts){{
  if(!lastTs) lastTs=ts;
  const dt=(ts-lastTs)/1000; lastTs=ts;
  if(running) simTime+=dt*speed;
  ctx.clearRect(0,0,W(),H());

  const active=flights.filter(f=>simTime>=f.depart_t&&simTime<=f.return_t+3);
  flights.forEach(f=>{{
    if(simTime>=f.depart_t-5)
      drawDest(destAngles[f.pkg_id], simTime>=f.depart_t&&simTime<f.return_t);
  }});

  const droneFlights={{}};
  active.forEach(f=>{{
    if(!(f.drone_id in droneFlights)||droneFlights[f.drone_id].depart_t<f.depart_t)
      droneFlights[f.drone_id]=f;
  }});

  let activeCount=0;
  const deliv=flights.filter(f=>simTime>f.arrive_t).length;

  droneState.forEach(d=>{{
    const f=droneFlights[d.id];
    if(f){{
      activeCount++;
      const angle=destAngles[f.pkg_id];
      const [dx,dy]=destXY(angle);
      const dur=f.arrive_t-f.depart_t;
      drawFlightPath(angle, COLORS[d.id%COLORS.length]);
      if(simTime<f.arrive_t){{
        const t=ease(Math.max(0,Math.min(1,(simTime-f.depart_t)/dur)));
        d.x=HX()+(dx-HX())*t; d.y=HY()+(dy-HY())*t;
        d.status='flying'; d.battery=Math.max(15,100-t*45);
      }} else {{
        const t=ease(Math.max(0,Math.min(1,(simTime-f.arrive_t)/dur)));
        d.x=dx+(HX()-dx)*t; d.y=dy+(HY()-dy)*t;
        d.status='returning'; d.battery=Math.max(10,55-t*25);
      }}
      d.trail.push({{x:d.x,y:d.y}});
      if(d.trail.length>28) d.trail.shift();
    }} else {{
      d.x=HX()+(Math.random()-.5)*3; d.y=HY()+(Math.random()-.5)*3;
      d.status='idle'; d.trail=[];
    }}
    drawDrone(d);
  }});

  drawHub();

  const clockMin = Math.round(SIM_START*60+simTime);
  const hh = String(Math.floor(clockMin/60)).padStart(2,'0');
  const mm = String(clockMin%60).padStart(2,'0');
  document.getElementById('timeLabel').textContent=`t = ${{Math.round(simTime)}} min  (${{hh}}:${{mm}})`;
  document.getElementById('activeCount').textContent=activeCount;
  document.getElementById('delivCount').textContent=deliv;

  if(simTime>930) simTime=0;
  requestAnimationFrame(frame);
}}
requestAnimationFrame(frame);
</script>
</body>
</html>"""

# sidebar
with st.sidebar:
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown(f"<div style='font-size:0.95rem;font-weight:600;color:{TEXT};padding:0.2rem 0'>drone hub sim</div>", unsafe_allow_html=True)
    with col_toggle:
        if st.button("🌙" if not dark else "☀️", key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f"<div class='notion-section'>fleet</div>", unsafe_allow_html=True)
    n_drones   = st.slider("drones", 3, 10, 6, 1)
    n_chargers = st.slider("charging stations", 1, 6, 2, 1)

    st.markdown(f"<div class='notion-section'>environment</div>", unsafe_allow_html=True)
    season_names      = list(WIND_SEASONS.keys())
    season_labels_map = {k: v['label'] for k, v in WIND_SEASONS.items()}
    selected_season   = st.selectbox(
        "season",
        options=season_names,
        format_func=lambda x: season_labels_map[x],
        index=0,
    )
    month = WIND_SEASONS[selected_season]['months'][0]

    st.markdown(f"<div class='notion-section'>experiment</div>", unsafe_allow_html=True)
    n_reps     = st.slider("replications", 5, 30, 10, 5)
    seed       = st.number_input("random seed", value=42, min_value=0, step=1)
    sla_target = st.slider("sla target (%)", 80, 99, 95, 1)

    wind_mult = WIND_SEASONS[selected_season]['mult']
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.7rem;background:{BG3};border-radius:6px;
         font-size:0.75rem;color:{TEXT2};line-height:2.1;">
      wind multiplier <span style="color:{TEXT};float:right">{wind_mult:.2f}×</span><br>
      sla window      <span style="color:{TEXT};float:right">120 min</span><br>
      dist range      <span style="color:{TEXT};float:right">3–8 km</span><br>
      sim day         <span style="color:{TEXT};float:right">07:00–22:00</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_single = st.button("Run simulation",  key="run_s")
    run_grid   = st.button("Run grid search", key="run_g")

# header
st.markdown(f"""
<div class='notion-title'>Last-Mile Drone Delivery</div>
<div class='notion-sub'>
  Discrete-event simulation · Accra, Ghana · DJI FlyCart 30
  <span class='tag tag-blue'>SimPy</span>
  <span class='tag tag-green'>{season_labels_map[selected_season]}</span>
  <span class='tag'>wind {wind_mult:.2f}×</span>
  <span class='tag'>{n_drones} drones · {n_chargers} chargers</span>
</div>
<hr class='notion-divider'>
""", unsafe_allow_html=True)

if 'single_results' not in st.session_state:
    st.session_state.single_results = None
if 'grid_results' not in st.session_state:
    st.session_state.grid_results   = None

if run_single:
    with st.spinner("running simulation..."):
        results, q_log, drones, flight_log = run_simulation(n_drones, n_chargers, seed, month)
    st.session_state.single_results = (results, q_log, drones, flight_log, n_drones, n_chargers, month)

if run_grid:
    drone_range   = list(range(3, 11))
    charger_range = list(range(1, 6))
    total_runs    = len(drone_range) * len(charger_range) * n_reps
    rows, done    = [], 0
    pb            = st.progress(0)
    for nd in drone_range:
        for nc in charger_range:
            for s in range(1, n_reps + 1):
                r, _, dr, _ = run_simulation(nd, nc, s, month)
                m = extract_metrics(r, dr, nd, nc, s, month)
                if m: rows.append(m)
                done += 1
                pb.progress(done / total_runs)
    raw_df  = pd.DataFrame(rows)
    summary = (
        raw_df.groupby(['n_drones', 'n_chargers'])
        .agg(
            on_time_mean     = ('on_time_rate',      'mean'),
            on_time_ci       = ('on_time_rate',      lambda x: stats.sem(x) * stats.t.ppf(0.975, df=len(x)-1)),
            p95_mean         = ('p95_delivery_t',    'mean'),
            late_mean        = ('late_count',        'mean'),
            util_fly_mean    = ('drone_util_fly',    'mean'),
            util_charge_mean = ('drone_util_charge', 'mean'),
            util_wait_mean   = ('drone_util_wait',   'mean'),
        )
        .reset_index()
    )
    summary['hits_sla']   = summary['on_time_mean'] >= (sla_target / 100)
    summary['cost_proxy'] = summary['n_drones'] * 3 + summary['n_chargers']
    st.session_state.grid_results = (raw_df, summary)
    pb.empty()

def kpi(col, label, value, sub='', cls=''):
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# single results
if st.session_state.single_results:
    results, q_log, drones, flight_log, nd, nc, mo = st.session_state.single_results
    df      = pd.DataFrame(results)
    metrics = extract_metrics(results, drones, nd, nc, seed, mo)
    on_time = metrics['on_time_rate']
    p95     = metrics['p95_delivery_t']

    st.markdown(f"<div class='notion-section'>results — {nd} drones · {nc} chargers · seed {seed}</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "on-time rate",    f"{on_time:.1%}",
        f"target {sla_target}%",
        'kpi-good' if on_time >= sla_target/100 else 'kpi-bad')
    kpi(c2, "packages",        f"{metrics['total_packages']}", f"late: {metrics['late_count']}")
    kpi(c3, "p95 delivery",    f"{p95:.0f} min",
        f"mean: {metrics['mean_delivery_t']:.0f} min",
        '' if p95 <= SLA_LIMIT else 'kpi-warn')
    kpi(c4, "avg wait",        f"{metrics['mean_wait_t']:.1f} min", f"p95: {metrics['p95_wait_t']:.1f} min")
    kpi(c5, "fleet util",      f"{metrics['drone_util_fly']*100:.0f}%",
        f"charging: {metrics['drone_util_charge']*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_anim, tab1, tab2, tab3, tab4 = st.tabs([
        "🚁  flight map", "delivery times", "queue dynamics", "drone utilisation", "raw data"
    ])

    with tab_anim:
        st.markdown(f"<div style='font-size:0.8rem;color:{TEXT2};margin-bottom:0.5rem'>animated drone flight playback — use the speed slider to scrub through the operating day</div>", unsafe_allow_html=True)
        components.html(build_flight_animation(flight_log, nd, dark), height=520, scrolling=False)

    with tab1:
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df['delivery_t'], nbinsx=40,
                marker_color=ACCENT, opacity=0.75, name='delivery time'))
            fig.add_vline(x=SLA_LIMIT, line_dash='dash', line_color=RED_C, line_width=1.5,
                          annotation_text='120 min SLA', annotation_font_color=RED_C)
            fig.update_layout(title='Delivery time distribution',
                              xaxis_title='minutes', yaxis_title='count')
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            sorted_t = np.sort(df['delivery_t'].values)
            cdf_vals = np.arange(1, len(sorted_t)+1) / len(sorted_t)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sorted_t, y=cdf_vals*100, mode='lines',
                line=dict(color=ACCENT, width=2), fill='tozeroy', fillcolor=hex_to_rgba(ACCENT, 0.1),
                name='CDF'))
            fig2.add_vline(x=SLA_LIMIT, line_dash='dash', line_color=RED_C, line_width=1.5)
            fig2.add_hline(y=sla_target, line_dash='dot', line_color=YELLOW_C, line_width=1.5)
            fig2.update_layout(title='CDF of delivery times',
                               xaxis_title='delivery time (min)', yaxis_title='cumulative %')
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        hours       = list(range(7, 22))
        lambda_vals = [LAMBDA_BY_HOUR.get(h, 0) for h in hours]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"{h:02d}:00" for h in hours], y=lambda_vals,
            marker_color=[ACCENT if l >= 8 else YELLOW_C if l >= 5 else GREEN_C for l in lambda_vals]))
        fig3.update_layout(title='Time-varying arrival rate λ',
                           xaxis_title='hour of day', yaxis_title='packages per hour')
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        if q_log:
            qdf  = pd.DataFrame(q_log)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=qdf['time'], y=qdf['q_len'], mode='lines', fill='tozeroy',
                line=dict(color=ACCENT, width=1.5), fillcolor=hex_to_rgba(ACCENT, 0.1),
                name='available drones'))
            fig4.update_layout(title='Available drones in pool over time',
                               xaxis_title='sim time (min)', yaxis_title='drones available')
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        util_data = []
        for d in drones:
            idle_t = max(0, SIM_DURATION - d.time_flying - d.time_charging - d.time_waiting)
            util_data.append({
                'drone':    f'D{d.id:02d}',
                'flying':   round(d.time_flying   / SIM_DURATION * 100, 1),
                'charging': round(d.time_charging / SIM_DURATION * 100, 1),
                'waiting':  round(d.time_waiting  / SIM_DURATION * 100, 1),
                'idle':     round(idle_t           / SIM_DURATION * 100, 1),
            })
        util_df = pd.DataFrame(util_data)
        fig5    = go.Figure()
        for cn, color, label in [
            ('flying', ACCENT, 'flying'), ('charging', GREEN_C, 'charging'),
            ('waiting', RED_C, 'waiting for charger'), ('idle', TEXT3, 'idle'),
        ]:
            fig5.add_trace(go.Bar(name=label, x=util_df['drone'], y=util_df[cn],
                                  marker_color=color, opacity=0.85))
        fig5.update_layout(barmode='stack', title='Drone time utilisation',
                           yaxis_title='% of sim duration', xaxis_title='')
        apply_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        disp = df[['pkg_id','arrival_t','delivery_t','wait_t','distance_km','is_heavy','late','drone_id']].copy()
        disp['arrival_t']  = disp['arrival_t'].round(1)
        disp['delivery_t'] = disp['delivery_t'].round(1)
        disp['wait_t']     = disp['wait_t'].round(1)
        st.dataframe(disp, use_container_width=True, height=380)

    st.markdown(f"<hr class='notion-divider'>", unsafe_allow_html=True)

# grid results
if st.session_state.grid_results:
    raw_df, summary = st.session_state.grid_results
    sla_configs     = summary[summary['hits_sla']].copy()
    optimal         = sla_configs.loc[sla_configs['cost_proxy'].idxmin()] if len(sla_configs) > 0 else None

    st.markdown(f"<div class='notion-section'>grid search — {n_reps} replications per config</div>", unsafe_allow_html=True)

    if optimal is not None:
        opt_d = int(optimal['n_drones'])
        opt_c = int(optimal['n_chargers'])
        gc1, gc2, gc3, gc4 = st.columns(4)
        kpi(gc1, "optimal config",  f"{opt_d}D + {opt_c}C",              f"cost proxy: {int(optimal['cost_proxy'])}", 'kpi-good')
        kpi(gc2, "on-time rate",    f"{optimal['on_time_mean']:.1%}",     f"±{optimal['on_time_ci']:.1%} CI")
        kpi(gc3, "p95 delivery",    f"{optimal['p95_mean']:.0f} min",     f"late/run: {optimal['late_mean']:.1f}")
        kpi(gc4, "configs passing", f"{len(sla_configs)}", f"of {len(summary)} tested")
    else:
        st.warning("no config met the SLA target — try adding more drones or chargers.")
        opt_d, opt_c = None, None

    st.markdown("<br>", unsafe_allow_html=True)
    gtab1, gtab2, gtab3 = st.tabs(["heatmap", "fleet size vs on-time", "utilisation breakdown"])

    with gtab1:
        heat_pivot = summary.pivot(index='n_drones', columns='n_chargers', values='on_time_mean') * 100
        cs = ([[0,'#3d1a19'],[0.5,'#1e3a2a'],[1,'#1e3a5f']] if dark else
              [[0,'#fce8e6'],[0.5,'#e9f5ee'],[1,'#e8f1fb']])
        fig_h = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=[f"{c} chargers" for c in heat_pivot.columns],
            y=[f"{d} drones"   for d in heat_pivot.index],
            colorscale=cs, zmin=50, zmax=100,
            text=np.round(heat_pivot.values, 1),
            texttemplate='%{text}%',
            textfont=dict(family='Inter', size=11, color=TEXT),
            colorbar=dict(title='on-time %', tickfont=dict(color=PLOT_TEXT)),
        ))
        if opt_d and opt_c:
            ox = list(heat_pivot.columns).index(opt_c)
            oy = list(heat_pivot.index).index(opt_d)
            fig_h.add_shape(type='rect', x0=ox-.5, x1=ox+.5, y0=oy-.5, y1=oy+.5,
                            line=dict(color=ACCENT, width=2.5, dash='dash'))
            fig_h.add_annotation(x=ox, y=oy-.65, text='optimal',
                font=dict(color=ACCENT, size=9), showarrow=False)
        fig_h.update_layout(title='On-time delivery rate by fleet configuration')
        apply_theme(fig_h, height=420)
        st.plotly_chart(fig_h, use_container_width=True)

    with gtab2:
        fig_l = go.Figure()
        for idx, nc_val in enumerate(sorted(summary['n_chargers'].unique())):
            sub   = summary[summary['n_chargers'] == nc_val].sort_values('n_drones')
            color = PALETTE[idx % len(PALETTE)]
            fig_l.add_trace(go.Scatter(
                x=sub['n_drones'], y=sub['on_time_mean']*100,
                mode='lines+markers', line=dict(color=color, width=2),
                marker=dict(size=6), name=f'{int(nc_val)} chargers',
                error_y=dict(type='data', array=(sub['on_time_ci']*100).values,
                             visible=True, color=color, thickness=1.2),
            ))
        fig_l.add_hline(y=sla_target, line_dash='dash', line_color=TEXT2, line_width=1.5,
                        annotation_text=f'{sla_target}% SLA', annotation_font_color=TEXT2)
        fig_l.update_layout(title='On-time rate vs fleet size',
                            xaxis_title='number of drones',
                            yaxis_title='on-time rate (%)', yaxis_range=[40, 105])
        apply_theme(fig_l)
        st.plotly_chart(fig_l, use_container_width=True)

    with gtab3:
        if opt_c is not None:
            # use summary aggregated columns to avoid the KeyError
            util_s = summary[summary['n_chargers'] == opt_c].sort_values('n_drones').copy()
            fig_u  = go.Figure()
            for col_key, color, label in [
                ('util_fly_mean',    ACCENT,  'flying'),
                ('util_charge_mean', GREEN_C, 'charging'),
                ('util_wait_mean',   RED_C,   'waiting for charger'),
            ]:
                fig_u.add_trace(go.Bar(
                    x=[f"{int(d)}D" for d in util_s['n_drones']],
                    y=util_s[col_key] * 100,
                    name=label, marker_color=color, opacity=0.85,
                ))
            fig_u.update_layout(
                barmode='stack',
                title=f'Drone utilisation at {opt_c} charging station{"s" if opt_c > 1 else ""}',
                xaxis_title='fleet size', yaxis_title='% of total drone time',
            )
            apply_theme(fig_u)
            st.plotly_chart(fig_u, use_container_width=True)

# noise model
with st.expander("noise model verification"):
    np.random.seed(0)
    dist_km        = 5.0
    drain_samples  = [DRAIN_BASE*dist_km + np.random.normal(0, PROCESS_NOISE_SD*np.sqrt(dist_km)) for _ in range(1000)]
    sensor_samples = [26.0 + np.random.normal(0, MEASURE_NOISE_SD) for _ in range(1000)]
    cn1, cn2 = st.columns(2)
    with cn1:
        fn1 = go.Figure()
        fn1.add_trace(go.Histogram(x=drain_samples, nbinsx=40, marker_color=ACCENT, opacity=0.7))
        fn1.add_vline(x=np.mean(drain_samples), line_dash='dash', line_color=YELLOW_C)
        fn1.update_layout(title=f'Battery drain noise — {dist_km}km trip',
                          xaxis_title='% drained', yaxis_title='count')
        apply_theme(fn1)
        st.plotly_chart(fn1, use_container_width=True)
    with cn2:
        fn2 = go.Figure()
        fn2.add_trace(go.Histogram(x=sensor_samples, nbinsx=40, marker_color=GREEN_C, opacity=0.7))
        fn2.add_vline(x=26.0, line_dash='dash', line_color=YELLOW_C,
                      annotation_text='true = 26%', annotation_font_color=YELLOW_C)
        fn2.update_layout(title='Battery sensor noise — true=26%',
                          xaxis_title='reported %', yaxis_title='count')
        apply_theme(fn2)
        st.plotly_chart(fn2, use_container_width=True)

st.markdown(f"""
<hr class='notion-divider'>
<div style="font-size:0.75rem;color:{TEXT3};display:flex;justify-content:space-between;padding-bottom:1rem">
  <span>group 2 · last-mile drone delivery hub simulation</span>
  <span>accra, ghana · DJI FlyCart 30 · simpy v4</span>
</div>
""", unsafe_allow_html=True)