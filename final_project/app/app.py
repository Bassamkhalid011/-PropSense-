"""
app.py  –  PropSense · Zameen Islamabad Price Predictor
=======================================================
Property Types: House ONLY
RUN:  streamlit run app.py
"""

import streamlit as st
import joblib, os, re, json
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PropSense · Islamabad",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
#  BlendModel MUST be defined here so joblib can unpickle it
# ══════════════════════════════════════════════════════════════
class BlendModel:
    def __init__(self, lgbm, xgb, w1, w2):
        self.lgbm = lgbm
        self.xgb  = xgb
        self.w1   = w1
        self.w2   = w2
    def predict(self, X):
        return self.w1 * self.lgbm.predict(X) + self.w2 * self.xgb.predict(X)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background:#0a0d14; color:#e8eaf2; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp { background: linear-gradient(135deg,#0a0d14 0%,#0e1220 50%,#0a0d14 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f1319,#111827); border-right: 1px solid rgba(99,170,255,0.12); }
.hero-banner { background: linear-gradient(135deg,#0f1b35,#0d2248 40%,#091a38); border: 1px solid rgba(99,170,255,0.2); border-radius: 20px; padding: 40px 48px; margin-bottom: 32px; position: relative; overflow: hidden; }
.hero-banner::before { content:''; position:absolute; top:-60px; right:-60px; width:280px; height:280px; background: radial-gradient(circle,rgba(99,170,255,0.12),transparent 70%); border-radius:50%; }
.hero-title { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:800; line-height:1.15; background: linear-gradient(90deg,#e8eaf2 30%,#63aaff 70%,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0 0 8px; }
.hero-sub { color:#6b7fa8; font-size:1rem; margin:0; letter-spacing:.02em; }
.stat-row { display:flex; gap:14px; margin-top:24px; flex-wrap:wrap; }
.stat-pill { background:rgba(99,170,255,0.08); border:1px solid rgba(99,170,255,0.18); border-radius:100px; padding:6px 18px; font-size:.82rem; color:#63aaff; font-weight:500; letter-spacing:.04em; }
.section-header { font-size:.7rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:#4a5878; margin:28px 0 14px; padding-bottom:8px; border-bottom:1px solid rgba(99,170,255,0.08); }
div[data-baseweb="select"] > div { background:#111827 !important; border:1px solid rgba(99,170,255,0.15) !important; border-radius:10px !important; color:#e8eaf2 !important; }
.stNumberInput input { background:#111827 !important; border:1px solid rgba(99,170,255,0.15) !important; border-radius:10px !important; color:#e8eaf2 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#2563eb,#4f46e5) !important; color:white !important; border:none !important; border-radius:14px !important; font-weight:600 !important; font-size:1rem !important; padding:16px !important; box-shadow:0 4px 24px rgba(37,99,235,0.35) !important; transition: all .2s !important; }
.stButton > button[kind="primary"]:hover { transform:translateY(-2px) !important; }
.result-card { background: linear-gradient(135deg,#0f2040,#0d1c38); border:1px solid rgba(99,170,255,0.25); border-radius:20px; padding:36px 40px; margin:24px 0; text-align:center; position:relative; overflow:hidden; }
.result-card::before { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 50% 0%,rgba(99,170,255,0.1),transparent 60%); }
.result-label { font-size:.75rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:#4a6898; margin-bottom:10px; }
.result-price { font-family:'Playfair Display',serif; font-size:3rem; font-weight:800; background:linear-gradient(90deg,#63aaff,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1; margin:0; }
.result-range { font-size:.88rem; color:#4a6898; margin-top:10px; }
.metric-card { background:#111827; border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:20px 22px; text-align:center; }
.metric-val { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#63aaff; }
.metric-lbl { font-size:.72rem; color:#4a5878; text-transform:uppercase; letter-spacing:.1em; margin-top:4px; }
.loc-card { background: linear-gradient(135deg,#0f1b35,#111827); border:1px solid rgba(99,170,255,0.2); border-radius:16px; padding:20px 24px; margin:12px 0; }
.loc-name { font-size:1rem; font-weight:600; color:#e8eaf2; margin-bottom:12px; }
.loc-metrics { display:flex; gap:20px; flex-wrap:wrap; }
.loc-m { flex:1; min-width:100px; }
.loc-m .val { font-size:1.2rem; font-weight:700; }
.loc-m .lbl { font-size:.68rem; color:#4a5878; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }
.badge-low  { background:rgba(220,38,38,0.15); color:#f87171; border:1px solid rgba(220,38,38,0.3); border-radius:8px; padding:2px 10px; font-size:.72rem; }
.badge-med  { background:rgba(37,99,235,0.15); color:#63aaff; border:1px solid rgba(37,99,235,0.3); border-radius:8px; padding:2px 10px; font-size:.72rem; }
.badge-high { background:rgba(5,150,105,0.15); color:#34d399; border:1px solid rgba(5,150,105,0.3); border-radius:8px; padding:2px 10px; font-size:.72rem; }
.accent-blue { color:#63aaff !important; }
.accent-purple { color:#a78bfa !important; }
.accent-green { color:#34d399 !important; }
.accent-amber { color:#fbbf24 !important; }
.stCheckbox label { color:#8b9cc5 !important; font-size:.9rem; }
.stTextInput label,.stNumberInput label,.stSelectbox label,.stSlider label { color:#6b7fa8 !important; font-size:.82rem !important; font-weight:500 !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0a0d14; }
::-webkit-scrollbar-thumb { background:#1e2a42; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────
def extract_sector(loc):
    if not loc: return "Unknown"
    m = re.search(r'\b([A-Z]-\d{1,2})\b', str(loc))
    if m: return m.group(1)
    parts = str(loc).split(",")
    return parts[-1].strip() if len(parts) > 1 else str(loc).strip()

def area_category(a):
    if a <= 6:  return '5_marla'
    if a <= 12: return '10_marla'
    if a <= 22: return '1_kanal'
    if a <= 42: return '2_kanal'
    return 'estate'

def safe_col(n):
    return re.sub(r'[^A-Za-z0-9_]', '_', n)

def fmt(pkr):
    if pkr >= 1e9:  return f"PKR {pkr/1e9:.2f} Arab"
    if pkr >= 1e7:  return f"PKR {pkr/1e7:.2f} Crore"
    if pkr >= 1e5:  return f"PKR {pkr/1e5:.2f} Lakh"
    return f"PKR {pkr:,.0f}"

def badge(count):
    if count < 5:  return '<span class="badge-low">⚠ Sparse</span>'
    if count < 20: return '<span class="badge-med">◉ Moderate</span>'
    return '<span class="badge-high">✓ Rich</span>'

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    a = {}
    a["model"]           = joblib.load(os.path.join(SCRIPT_DIR, "best_model.pkl"))
    a["expected_cols"]   = joblib.load(os.path.join(SCRIPT_DIR, "expected_cols.pkl"))
    a["le_type"]         = joblib.load(os.path.join(SCRIPT_DIR, "property_type_le.pkl"))
    a["loc_map"]         = joblib.load(os.path.join(SCRIPT_DIR, "loc_mean_map.pkl"))
    a["sector_map"]      = joblib.load(os.path.join(SCRIPT_DIR, "sector_mean_map.pkl"))
    a["global_mean"]     = joblib.load(os.path.join(SCRIPT_DIR, "global_mean.pkl"))
    a["loc_stats"]       = joblib.load(os.path.join(SCRIPT_DIR, "loc_stats_dict.pkl"))
    a["top_locs_ohe"]    = joblib.load(os.path.join(SCRIPT_DIR, "top_locs_ohe.pkl"))
    a["imputer"]         = joblib.load(os.path.join(SCRIPT_DIR, "knn_imputer.pkl"))
    a["sector_ppm"]      = joblib.load(os.path.join(SCRIPT_DIR, "sector_ppm.pkl"))
    a["sector_area_ppm"] = joblib.load(os.path.join(SCRIPT_DIR, "sector_area_ppm.pkl"))
    a["luxury_map"]      = joblib.load(os.path.join(SCRIPT_DIR, "luxury_sector_map.pkl"))
    a["area_mult"]       = joblib.load(os.path.join(SCRIPT_DIR, "area_multipliers.pkl"))
    a["global_ppm"]      = joblib.load(os.path.join(SCRIPT_DIR, "global_ppm.pkl"))
    with open(os.path.join(SCRIPT_DIR, "top_locations.txt")) as f:
        a["all_locations"] = sorted(f.read().splitlines())
    with open(os.path.join(SCRIPT_DIR, "model_metrics.json")) as f:
        a["metrics"] = json.load(f)
    return a

assets = load_assets()
m    = assets["metrics"]
best = m["best"]
bm   = m["metrics"][best]

# ── Dynamic PPM ───────────────────────────────────────────────
def get_dynamic_ppm(sector, area):
    ac  = area_category(area)
    sap = assets["sector_area_ppm"]
    spm = assets["sector_ppm"]
    am  = assets["area_mult"]
    gp  = assets["global_ppm"]
    if sector in sap and ac in sap[sector] and sap[sector][ac]['reliable']:
        return float(sap[sector][ac]['ppm'])
    if sector in spm:
        return float(spm[sector] * am.get(ac, 1.0))
    return float(gp * am.get(ac, 1.0))

# ══════════════════════════════════════════════════════════════
#  build_features — works with ANY model saved by the notebook
#  Builds ALL possible features, then selects only what model needs
# ══════════════════════════════════════════════════════════════
def build_features(area, beds, baths, parking, servant_q, store_rooms,
                   kitchens, prop_age, is_new, location, parking_known, servant_known):
    expected = assets["expected_cols"]
    sector     = extract_sector(location)
    loc_enc    = float(assets["loc_map"].get(location, assets["global_mean"]))
    sector_enc = float(assets["sector_map"].get(sector, assets["global_mean"]))
    dyn_ppm    = get_dynamic_ppm(sector, area)
    baseline   = dyn_ppm * area
    ac         = area_category(area)
    ac_enc     = float({'5_marla':0,'10_marla':1,'1_kanal':2,'2_kanal':3,'estate':4}.get(ac, 0))

    # Build ALL features (superset)
    all_feats = {
        # Raw inputs
        'area':             float(area),
        'bedrooms':         float(beds),
        'bathrooms':        float(baths),
        'parking':          float(parking),
        'servant_quarters': float(servant_q),
        'store_rooms':      float(store_rooms),
        'kitchens':         float(kitchens),
        'property_age':     float(prop_age),
        'is_new':           float(is_new),
        'drawing_rooms':    0.0,
        # Known flags (old model compatibility)
        'parking_known':          float(parking_known),
        'servant_quarters_known': float(servant_known),
        'store_rooms_known':      1.0,
        'kitchens_known':         1.0,
        'drawing_rooms_known':    0.0,
        # Encodings
        'location_enc':     loc_enc,
        'sector_enc':       sector_enc,
        'log_loc_price':    np.log1p(loc_enc),
        'log_sector_price': np.log1p(sector_enc),
        # Pricing intelligence features
        'dynamic_ppm':      dyn_ppm,
        'baseline_price':   baseline,
        'log_baseline':     np.log1p(baseline),
        'log_dynamic_ppm':  np.log1p(dyn_ppm),
        'is_luxury_sector': float(sector in assets["luxury_map"]),
        'sector_ppm_enc':   float(assets["sector_ppm"].get(sector, assets["global_ppm"])),
        'luxury_premium':   float(assets["luxury_map"].get(sector, 1.0)),
        'area_cat_enc':     ac_enc,
        'size_class':       ac_enc,
        'ppm_x_area':       dyn_ppm * area,
        'ppm_x_beds':       dyn_ppm * beds,
        'baseline_x_beds':  np.log1p(baseline) * beds,
        # Engineered features
        'log_area':          np.log1p(area),
        'log_beds':          np.log1p(beds),
        'log_baths':         np.log1p(baths),
        'area_sq':           area ** 2,
        'log_area_sq':       np.log1p(area) ** 2,
        'beds_x_area':       beds * area,
        'beds_x_log_area':   beds * np.log1p(area),
        'bath_bed_ratio':    min(baths / max(beds, 1), 5),
        'total_rooms':       beds + baths + kitchens + store_rooms,
        'log_total_rooms':   np.log1p(beds + baths + kitchens + store_rooms),
        'amenity_score':     int(parking > 0) + int(servant_q > 0) + int(store_rooms > 0),
        'price_per_marla_loc': loc_enc / max(area, 1),
        'loc_x_area':        loc_enc * np.log1p(area),
        'loc_x_beds':        loc_enc * beds,
        'loc_x_size_class':  loc_enc * ac_enc,
        'luxury_score':      beds * baths * np.log1p(area),
        'log_luxury_score':  np.log1p(beds * baths * np.log1p(area)),
        'age_x_loc':         float(prop_age) * np.log1p(loc_enc),
        # Old model features
        'log_area':          np.log1p(area),
        'log_beds':          np.log1p(beds),
        'log_sector_ppm':    float(assets["sector_ppm"].get(sector, assets["global_ppm"])),
        'is_new_listing':    float(is_new),
    }

    # Property type encoding
    try:
        all_feats['property_type_enc'] = float(assets["le_type"].transform(['House'])[0])
    except:
        all_feats['property_type_enc'] = 0.0

    # OHE location columns
    loc_col = safe_col(f'loc_{location}')
    all_feats[loc_col] = 1.0
    # Also try old OHE format
    old_loc_col = f'loc_ohe_{location}'
    all_feats[safe_col(old_loc_col)] = 1.0

    # Build final DataFrame using ONLY columns the model expects
    row = {c: all_feats.get(c, 0.0) for c in expected}
    return pd.DataFrame([row])[expected].astype(float)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(90deg,#63aaff,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>PropSense</div>
      <div style='font-size:.72rem;color:#3d4f6e;letter-spacing:.12em;text-transform:uppercase;margin-top:2px;'>
          Islamabad Real Estate AI</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🔮  Predict Price",
        "📍  Location Prices",
        "📊  Model Dashboard",
        "🗺️  Market Overview",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:rgba(99,170,255,0.1);margin:20px 0;'></div>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:#111827;border:1px solid rgba(99,170,255,0.12);border-radius:12px;padding:16px 18px;'>
      <div style='color:#63aaff;font-weight:600;font-size:.9rem;margin-bottom:4px;'>Active Model</div>
      <div style='color:#e8eaf2;font-size:.85rem;font-weight:500;'>{best}</div>
      <div style='display:flex;gap:16px;margin-top:12px;'>
        <div>
          <div style='color:#e8eaf2;font-weight:700;font-size:1.1rem;'>{bm["R2"]:.4f}</div>
          <div style='color:#3d4f6e;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;'>R² Score</div>
        </div>
        <div>
          <div style='color:#34d399;font-weight:700;font-size:1.1rem;'>{bm["MAE"]/1e5:.1f}L</div>
          <div style='color:#3d4f6e;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;'>Avg Error</div>
        </div>
        <div>
          <div style='color:#fbbf24;font-weight:700;font-size:1.1rem;'>{bm.get("Accuracy",0)*100:.1f}%</div>
          <div style='color:#3d4f6e;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;'>Accuracy</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 – PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🔮  Predict Price":
    st.markdown(f"""
    <div class="hero-banner">
      <p class="hero-title">Predict Property<br>Value Instantly</p>
      <p class="hero-sub">AI-powered pricing for Houses in Islamabad · {best}</p>
      <div class="stat-row">
        <span class="stat-pill">{bm['R2']*100:.1f}% R² Accuracy</span>
        <span class="stat-pill">{bm.get('Accuracy',0)*100:.1f}% Classification</span>
        <span class="stat-pill">🏠 Houses Only</span>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1], gap="large")

    with c1:
        st.markdown('<div class="section-header">📐 Property Details</div>', unsafe_allow_html=True)
        area     = st.number_input("Area (Marla) — 1 Kanal = 20 Marla", 1.0, 2000.0, 10.0, 0.5)
        beds     = st.slider("Bedrooms",  1, 15, 4)
        baths    = st.slider("Bathrooms", 1, 15, 3)
        is_new   = st.checkbox("New / Under-construction", value=False)
        prop_age = st.number_input("Property Age (years)", 0, 75,
                                    0 if is_new else 5, disabled=bool(is_new))
        if is_new: prop_age = 0

    with c2:
        st.markdown('<div class="section-header">📍 Type & Location</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:rgba(37,99,235,0.1);border:1px solid rgba(37,99,235,0.3);border-radius:10px;padding:10px 16px;color:#63aaff;font-weight:600;margin-bottom:12px;">🏠 House</div>',
                    unsafe_allow_html=True)
        loc_list    = assets["all_locations"]
        default_idx = loc_list.index("F-6, Islamabad") if "F-6, Islamabad" in loc_list else 0
        location    = st.selectbox("Location", loc_list, index=default_idx)
        kitchens    = st.slider("Kitchens",    0, 5, 1)
        store_rooms = st.slider("Store Rooms", 0, 5, 0)

        # Live location info + dynamic PPM
        sector  = extract_sector(location)
        dyn_ppm = get_dynamic_ppm(sector, area)
        loc_info = assets["loc_stats"].get(location, {})
        cnt = int(loc_info.get('listing_count', 0)) if loc_info else 0
        avg = loc_info.get('smoothed_mean', assets["global_mean"]) if loc_info else assets["global_mean"]
        is_lux = sector in assets["luxury_map"]
        st.markdown(f"""
        <div style='background:rgba(99,170,255,0.06);border:1px solid rgba(99,170,255,0.15);
                    border-radius:10px;padding:14px 16px;margin-top:10px;'>
          <div style='font-size:.68rem;color:#4a5878;text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:6px;'>Location Intelligence</div>
          <div style='font-size:1.2rem;font-weight:700;color:#63aaff;'>PKR {avg/1e7:.2f} Crore avg</div>
          <div style='font-size:.8rem;color:#6b7fa8;margin-top:4px;'>
              PPM: <b style="color:#34d399;">PKR {dyn_ppm/1e5:.1f}L/Marla</b>
              &nbsp;·&nbsp; {cnt} listings {badge(cnt)}
              {'&nbsp;·&nbsp;<span style="color:#fbbf24;">⭐ Luxury</span>' if is_lux else ''}
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-header">🏠 Amenities</div>', unsafe_allow_html=True)
        parking_known = st.checkbox("Know parking count?", value=True)
        parking       = st.slider("Parking Spaces",   0, 10, 1, disabled=not parking_known)
        servant_known = st.checkbox("Know servant quarters?", value=True)
        servant_q     = st.slider("Servant Quarters", 0, 5,  0, disabled=not servant_known)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✦  Calculate Property Value", type="primary", use_container_width=True):
        X = build_features(
            area=area, beds=beds, baths=baths,
            parking=parking if parking_known else 0,
            servant_q=servant_q if servant_known else 0,
            store_rooms=store_rooms, kitchens=kitchens,
            prop_age=prop_age, is_new=int(is_new),
            location=location,
            parking_known=int(parking_known),
            servant_known=int(servant_known),
        )
        pred_pkr = np.expm1(assets["model"].predict(X)[0])
        low_pkr  = pred_pkr * 0.88
        high_pkr = pred_pkr * 1.12
        baseline = get_dynamic_ppm(extract_sector(location), area) * area

        st.balloons()
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Estimated Market Value · 🏠 House</div>
          <div class="result-price">{fmt(pred_pkr)}</div>
          <div class="result-range">
            Range &nbsp;·&nbsp; {fmt(low_pkr)} &nbsp;–&nbsp; {fmt(high_pkr)}
            &nbsp;&nbsp;|&nbsp;&nbsp; Sector baseline: {fmt(baseline)}
          </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl, cls in [
            (c1, best,                        "Model",     "accent-blue"),
            (c2, f"{bm['R2']*100:.1f}%",      "R² Score",  "accent-purple"),
            (c3, f"PKR {dyn_ppm/1e5:.1f}L/M", "PPM",       "accent-green"),
            (c4, f"{area}M = {area/20:.1f}K",  "Area",     "accent-amber"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-val {cls}">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        CHARTS = os.path.join(SCRIPT_DIR, "charts")
        st.markdown('<div class="section-header" style="margin-top:32px;">📈 Model Insights</div>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        fi = os.path.join(CHARTS, "feature_importance.png")
        av = os.path.join(CHARTS, "actual_vs_predicted.png")
        if os.path.exists(fi): ca.image(fi, caption="Feature Importance", use_column_width=True)
        if os.path.exists(av): cb.image(av, caption="Actual vs Predicted", use_column_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 – LOCATION PRICES
# ══════════════════════════════════════════════════════════════
elif page == "📍  Location Prices":
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">Location-wise<br>Price Intelligence</p>
      <p class="hero-sub">Dynamic PPM · Mean prices · Data confidence per location</p>
    </div>""", unsafe_allow_html=True)

    col_s, col_f = st.columns([2,1])
    with col_s:
        search  = st.text_input("🔍 Search location or sector",
                                 placeholder="e.g. F-7, DHA, Bahria, G-11...")
    with col_f:
        sort_by = st.selectbox("Sort by", ["Mean Price ↓","Mean Price ↑",
                                            "Most Listings","Least Listings"])

    rows = []
    for loc, info in assets["loc_stats"].items():
        sec    = extract_sector(loc)
        ac_ppm = get_dynamic_ppm(sec, 10)
        rows.append({
            "location":      loc,
            "sector":        sec,
            "mean_price":    info.get("mean_price", 0),
            "smoothed_mean": info.get("smoothed_mean", 0),
            "median_price":  info.get("median_price", 0),
            "listing_count": int(info.get("listing_count", 0)),
            "ref_ppm":       ac_ppm,
            "is_luxury":     sec in assets["luxury_map"],
        })
    df_loc = pd.DataFrame(rows)

    if search:
        df_loc = df_loc[df_loc["location"].str.contains(search, case=False, na=False)]

    if sort_by == "Mean Price ↓":   df_loc = df_loc.sort_values("smoothed_mean", ascending=False)
    elif sort_by == "Mean Price ↑": df_loc = df_loc.sort_values("smoothed_mean", ascending=True)
    elif sort_by == "Most Listings": df_loc = df_loc.sort_values("listing_count", ascending=False)
    else: df_loc = df_loc.sort_values("listing_count", ascending=True)

    sparse = (df_loc["listing_count"] < 5).sum()
    lux    = df_loc["is_luxury"].sum()
    st.markdown(f"""
    <div style='display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;'>
      <div class="metric-card" style='flex:1;min-width:130px;'>
        <div class="metric-val accent-blue">{len(df_loc)}</div><div class="metric-lbl">Locations</div>
      </div>
      <div class="metric-card" style='flex:1;min-width:130px;'>
        <div class="metric-val accent-amber">{sparse}</div><div class="metric-lbl">Sparse (&lt;5)</div>
      </div>
      <div class="metric-card" style='flex:1;min-width:130px;'>
        <div class="metric-val accent-green">PKR {df_loc["smoothed_mean"].median()/1e7:.2f}Cr</div>
        <div class="metric-lbl">Median Price</div>
      </div>
      <div class="metric-card" style='flex:1;min-width:130px;'>
        <div class="metric-val" style="color:#fbbf24;">{lux}</div>
        <div class="metric-lbl">Luxury Sectors</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">All Locations</div>', unsafe_allow_html=True)
    for _, row in df_loc.head(100).iterrows():
        cnt     = row["listing_count"]
        mean    = row["smoothed_mean"]
        med     = row["median_price"]
        ppm     = row["ref_ppm"]
        lux_tag = ' &nbsp;<span style="color:#fbbf24;font-size:.75rem;">⭐ Luxury</span>' if row["is_luxury"] else ""
        sm_note = " <span style='color:#4a5878;font-size:.7rem;'>(smoothed)</span>" if cnt < 5 else ""
        st.markdown(f"""
        <div class="loc-card">
          <div class="loc-name">{row['location']} {badge(cnt)}{lux_tag}</div>
          <div class="loc-metrics">
            <div class="loc-m"><div class="val accent-blue">PKR {mean/1e7:.2f} Cr</div><div class="lbl">Avg Price{sm_note}</div></div>
            <div class="loc-m"><div class="val accent-purple">PKR {med/1e7:.2f} Cr</div><div class="lbl">Median</div></div>
            <div class="loc-m"><div class="val accent-green">PKR {ppm/1e5:.1f}L</div><div class="lbl">PPM (10M ref)</div></div>
            <div class="loc-m"><div class="val accent-amber">{cnt}</div><div class="lbl">Listings</div></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 – MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Dashboard":
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">Model Performance<br>Dashboard</p>
      <p class="hero-sub">All ML algorithms compared — Houses · Islamabad</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f2040,#1a1040);
                border:1px solid rgba(167,139,250,0.25);border-radius:16px;
                padding:24px 28px;margin-bottom:24px;'>
      <div style='font-size:.72rem;color:#4a5878;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;'>🏆 Winner</div>
      <div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap;'>
        <div style='font-family:Playfair Display,serif;font-size:1.8rem;font-weight:800;color:#a78bfa;'>{best}</div>
        <div style='color:#34d399;font-size:1.1rem;font-weight:600;'>R² {bm["R2"]:.4f}</div>
        <div style='color:#63aaff;font-size:1.1rem;font-weight:600;'>MAE {bm["MAE"]/1e5:.1f}L PKR</div>
        <div style='color:#fbbf24;font-size:1.1rem;font-weight:600;'>Acc {bm.get("Accuracy",0)*100:.1f}%</div>
      </div>
    </div>""", unsafe_allow_html=True)

    rows = []
    for name, v in sorted(m["metrics"].items(), key=lambda x: x[1]["R2"], reverse=True):
        rows.append({
            "Model":       name,
            "R² Score":    round(v["R2"], 4),
            "MAE (Lakh)":  round(v["MAE"] / 1e5, 1),
            "RMSE (Lakh)": round(v["RMSE"] / 1e5, 1),
            "Accuracy":    round(v.get("Accuracy", 0), 4),
            "F1 Score":    round(v.get("F1", 0), 4),
            "🏆":          "✅" if name == best else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    CHARTS = os.path.join(SCRIPT_DIR, "charts")
    for chart, caption in [
        ("model_comparison.png",   "Model R² Comparison"),
        ("feature_importance.png", "Feature Importance"),
        ("actual_vs_predicted.png","Actual vs Predicted"),
        ("error_distribution.png", "Error Distribution"),
        ("sector_ppm.png",         "Sector PPM Intelligence"),
    ]:
        path = os.path.join(CHARTS, chart)
        if os.path.exists(path):
            st.markdown(f'<div class="section-header">{caption}</div>', unsafe_allow_html=True)
            st.image(path, use_column_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 – MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════
elif page == "🗺️  Market Overview":
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">Islamabad Market<br>Overview</p>
      <p class="hero-sub">House listings — Live stats from Zameen.com</p>
    </div>""", unsafe_allow_html=True)

    data_path = os.path.join(SCRIPT_DIR, "../data/zameen_islamabad.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        def cp(v):
            if pd.isna(v): return np.nan
            v = str(v).lower().replace(",","").strip()
            try:
                n = float(v.split()[0])
                if "lakh"  in v: return n * 1e5
                if "crore" in v: return n * 1e7
                if "arab"  in v: return n * 1e9
                return n
            except: return np.nan
        df["price_clean"] = df["price"].apply(cp)
        df = df[df["property_type"] == "House"]
        df = df.dropna(subset=["price_clean","location"])

        c1,c2,c3,c4 = st.columns(4)
        for col, val, lbl, cls in [
            (c1, f"{len(df):,}",                              "House Listings",   "accent-blue"),
            (c2, f"{df['price_clean'].mean()/1e7:.2f} Cr",    "Avg Price",        "accent-purple"),
            (c3, f"{df['price_clean'].median()/1e7:.2f} Cr",  "Median Price",     "accent-green"),
            (c4, str(df['location'].nunique()),                "Unique Locations", "accent-amber"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-val {cls}">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:32px;">Top 25 Locations by Avg Price</div>',
                    unsafe_allow_html=True)
        avg = df.groupby("location")["price_clean"].mean().sort_values(ascending=False).head(25)
        st.bar_chart(pd.DataFrame({"Avg Price (Crore PKR)": avg.values/1e7}, index=avg.index))

        CHARTS = os.path.join(SCRIPT_DIR, "charts")
        for chart, caption in [("eda_overview.png","EDA Overview"),("sector_ppm.png","Sector PPM")]:
            path = os.path.join(CHARTS, chart)
            if os.path.exists(path):
                st.markdown(f'<div class="section-header">{caption}</div>', unsafe_allow_html=True)
                st.image(path, use_column_width=True)
    else:
        st.warning("Dataset not found. Run scraper first.")