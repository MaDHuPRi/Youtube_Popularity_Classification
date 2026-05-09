import streamlit as st
import pandas as pd
import numpy as np
import isodate
import re
import ast
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Download NLTK data ─────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0A0A0F !important; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F0F18 !important;
    border-right: 1px solid #1E1E30 !important;
}
[data-testid="stSidebar"] * { color: #C8C8E0 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background: #1A1A28 !important;
    border: 1px solid #2E2E48 !important;
    border-radius: 8px !important;
    color: #E0E0F0 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #1A1A28 !important;
    border: 1px solid #2E2E48 !important;
    border-radius: 8px !important;
    color: #E0E0F0 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #E0E0F0 !important; }
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6060A0 !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF3366, #FF6B35) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.75rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-size: 0.95rem !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Main text ── */
.main p, .main li { color: #C8C8E0 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #C8C8E0 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #E8E8FF !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E1E30 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #5050A0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.25rem !important;
}
.stTabs [aria-selected="true"] {
    color: #FF3366 !important;
    border-bottom: 2px solid #FF3366 !important;
}

/* ── Cards ── */
.card {
    background: #0F0F18;
    border: 1px solid #1E1E30;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
}
.card * { color: #C8C8E0 !important; }

/* ── Result boxes ── */
.result-viral {
    background: linear-gradient(135deg, #1a0020, #2a0030);
    border: 2px solid #FF3366;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #1a1000, #2a1800);
    border: 2px solid #FF6B35;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, #001a10, #002018);
    border: 2px solid #00C87A;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #0a0a1a, #10101e);
    border: 2px solid #4040A0;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

/* ── Stat chip ── */
.chip {
    display: inline-block;
    background: #1A1A28;
    border: 1px solid #2E2E48;
    color: #A0A0D0 !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin: 0.2rem;
}

/* ── Hero ── */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    color: #E8E8FF;
    line-height: 1;
    letter-spacing: 0.02em;
}
.hero-accent { color: #FF3366; }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4040A0;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}

/* ── Info ── */
.info-box {
    background: #0F0F18;
    border: 1px solid #1E1E30;
    border-left: 3px solid #FF3366;
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    margin-bottom: 1rem;
    color: #A0A0D0 !important;
    font-size: 0.88rem;
}
.info-box * { color: #A0A0D0 !important; }

/* ── Metric ── */
.metric-block {
    background: #0F0F18;
    border: 1px solid #1E1E30;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #FF3366;
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4040A0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF3366, #FF6B35);
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.7rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHANNELS = [
    '5mincrafts', 'Bright_side', 'Buzzfeed', 'Tasty_6000',
    'Thu_vu', 'averysmith', 'grateandgarnish5877', 'thefitnessmarshall',
    'tina_yong', 'VickyZhaoBEEAMP', 'TechwithLucy', 'tressuni',
    'itgirltierra', 'DarshilParmar', 'jayzern'
]
CHANNEL_OHE_COLS = [f'channel_{c}' for c in CHANNELS]

BASE_FEATURES = [
    'title_length', 'title_sentiment', 'has_description',
    'desc_sentiment', 'tag_count', 'duration_sec', 'hour'
]
REGRESSION_FEATURES = BASE_FEATURES + CHANNEL_OHE_COLS
CLUSTER_FEATURES = BASE_FEATURES + ['log_viewCount']

# ── Load & train model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load CSV, engineer features, train regression + clustering models."""
    import os, pathlib
    cwd = pathlib.Path(os.getcwd())
    # Load split CSV parts and combine
    filenames = ["merged_1_1.csv", "merged_1_2.csv", "merged_2_1.csv", "merged_2_2.csv"]
    parts = []
    for name in filenames:
        part_path = cwd / "data" / name
        if part_path.exists():
            parts.append(pd.read_csv(part_path))
    if not parts:
        st.error("No CSV parts found. Place merged_1_1.csv, merged_1_2.csv, merged_2_1.csv, merged_2_2.csv in the data/ folder.")
        return None, None, None, None, None, None
    df = pd.concat(parts, ignore_index=True)

    # ── Feature engineering ──
    df.drop_duplicates(inplace=True)
    df.drop(columns=[c for c in ['caption', 'favoriteCount'] if c in df.columns], inplace=True)

    for col in ['viewCount', 'likeCount', 'commentCount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df.dropna(subset=['publishedAt'], inplace=True)
        df['hour'] = df['publishedAt'].dt.hour
    else:
        df['hour'] = 12

    if 'tags' in df.columns:
        def safe_parse_tags(x):
            if pd.isnull(x) or x == '': return []
            try:
                r = ast.literal_eval(x)
                return r if isinstance(r, list) else [str(r)]
            except: return [str(x)]
        df['tags'] = df['tags'].apply(safe_parse_tags)
        df['tag_count'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df['tag_count'] = 0

    # Duration
    def safe_parse(x):
        try: return isodate.parse_duration(str(x)).total_seconds()
        except: return None
    if 'duration' in df.columns:
        df = df[~df['duration'].astype(str).str.lower().isin(['duration', 'nan'])]
        df['duration_sec'] = df['duration'].apply(safe_parse)
        df = df[df['duration_sec'].notna()]
    else:
        df['duration_sec'] = 300

    # Sentiment
    sid_model = SentimentIntensityAnalyzer()
    df['title'] = df['title'].fillna('').astype(str) if 'title' in df.columns else ''
    df['description'] = df['description'].fillna('').astype(str) if 'description' in df.columns else ''
    df['title_length'] = df['title'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['has_description'] = df['description'].apply(lambda x: 1 if isinstance(x, str) and x.strip() else 0)
    df['title_sentiment'] = df['title'].apply(lambda x: sid_model.polarity_scores(str(x) if pd.notna(x) else '')['compound'])
    df['desc_sentiment'] = df['description'].apply(lambda x: sid_model.polarity_scores(str(x) if pd.notna(x) else '')['compound'])

    # Medians for null fill
    for col in ['likeCount', 'viewCount', 'commentCount']:
        if col in df.columns:
            df[col].fillna(round(df[col].median()), inplace=True)

    # Channel OHE
    if 'channelTitle' in df.columns:
        channel_dummies = pd.get_dummies(df['channelTitle'], prefix='channel')
        df = pd.concat([df, channel_dummies], axis=1)

    # Ensure all OHE cols exist
    for col in CHANNEL_OHE_COLS:
        if col not in df.columns:
            df[col] = 0

    # Log view count
    if 'viewCount' in df.columns:
        df['log_viewCount'] = np.log1p(df['viewCount'].astype(float))
    else:
        return None, None, None, None, None, None

    # ── Train regression ──
    # Drop rows with NaN in any feature or target column
    train_cols = REGRESSION_FEATURES + ['log_viewCount']
    df = df.dropna(subset=train_cols)
    # Replace any remaining inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=train_cols)
    # Ensure all feature columns are numeric
    for col in REGRESSION_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=train_cols)

    X = df[REGRESSION_FEATURES]
    y = df['log_viewCount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ── Train KMeans clustering ──
    X_cluster = df[CLUSTER_FEATURES]
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, df['kmeans_cluster'])

    return pipeline, kmeans, df, r2, rmse, sil


def get_cluster_label(kmeans_model, cluster_id):
    centers = kmeans_model.cluster_centers_[:, 6]  # log_viewCount column
    sorted_indices = centers.argsort()
    labels = ['Low', 'Medium', 'High', 'Viral']
    mapping = {int(cluster_idx): labels[i] for i, cluster_idx in enumerate(sorted_indices)}
    return mapping.get(int(cluster_id), "Unknown")


def build_sample(title, tags_count, description, duration_str, hour, channel_name, sid_model):
    title_length = len(title)
    has_description = int(bool(description.strip()))
    title_senti = sid_model.polarity_scores(title)['compound']
    desc_senti = sid_model.polarity_scores(description)['compound']
    try:
        duration_sec = isodate.parse_duration(duration_str).total_seconds()
    except:
        duration_sec = 0

    base = pd.DataFrame([{
        'title_length': title_length,
        'title_sentiment': title_senti,
        'has_description': has_description,
        'desc_sentiment': desc_senti,
        'tag_count': tags_count,
        'duration_sec': duration_sec,
        'hour': hour,
    }])
    ohe = pd.DataFrame({col: [1 if col == f'channel_{channel_name}' else 0] for col in CHANNEL_OHE_COLS})
    sample = pd.concat([base, ohe], axis=1)[REGRESSION_FEATURES]
    return sample, duration_sec, title_senti, desc_senti


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem;'>
        <div style='font-size:2rem;'>🎥</div>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.6rem; color:#E8E8FF; letter-spacing:0.05em; line-height:1;'>YouTube<br>Predictor</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.6rem; color:#4040A0; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.4rem;'>Regression + Clustering</div>
    </div>
    <hr style='border-color:#1E1E30; margin:1rem 0;'>
    <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#4040A0; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.75rem;'>Video Details</div>
    """, unsafe_allow_html=True)

    title_input = st.text_input("Video Title", value="5 Hacks to Save Your Day!")
    tags_input = st.number_input("Number of Tags", min_value=0, max_value=100, value=10)
    desc_input = st.text_area("Description", value="These are some amazing life hacks that will surprise you.", height=80)
    duration_input = st.text_input("Duration (ISO format)", value="PT5M30S", help="e.g. PT5M30S = 5 min 30 sec")
    hour_input = st.slider("Hour of Upload (24h)", 0, 23, 12)
    channel_input = st.selectbox("Channel Name", CHANNELS)

    st.markdown("<hr style='border-color:#1E1E30; margin:0.75rem 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Popularity")

    st.markdown("""
    <hr style='border-color:#1E1E30; margin:1rem 0 0.75rem;'>
    <div style='font-family:"DM Mono",monospace; font-size:0.58rem; color:#303060; text-align:center; line-height:1.8;'>
        GradientBoosting + KMeans<br>
        VADER Sentiment · ISO Duration<br>
        © 2025
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>YouTube <span class='hero-accent'>Popularity</span><br>Predictor</div>
<div class='hero-sub'>// Gradient Boosting · KMeans Clustering · VADER Sentiment</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
sid = download_nltk()

with st.spinner("Training models on your dataset…"):
    reg_model, kmeans_model, df_trained, model_r2, model_rmse, model_sil = load_models()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predict", "📊  Model Performance", "🗂️  Data Explorer"])

# ══════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════
with tab1:
    if reg_model is None:
        st.markdown("""
        <div class='info-box'>
            ⚠️ <b>Dataset not found.</b> Place <code>merged.csv</code> in the <code>data/</code> folder next to <code>app.py</code> and restart.
        </div>
        """, unsafe_allow_html=True)
    elif not predict_btn:
        st.markdown("""
        <div class='info-box'>
            🎬 Fill in your video details in the sidebar and click <b>🔮 Predict Popularity</b> to get a view count prediction and popularity category.
        </div>
        """, unsafe_allow_html=True)

        # Model summary chips
        if model_r2:
            st.markdown(f"""
            <div style='margin-top:0.5rem;'>
                <span class='chip'>R² {model_r2:.3f}</span>
                <span class='chip'>RMSE {model_rmse:.3f}</span>
                <span class='chip'>Silhouette {model_sil:.3f}</span>
                <span class='chip'>GradientBoosting · KMeans-4</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        sample, dur_sec, title_senti, desc_senti = build_sample(
            title_input, tags_input, desc_input,
            duration_input, hour_input, channel_input, sid
        )

        log_pred = reg_model.predict(sample)[0]
        view_pred = int(np.expm1(log_pred))

        cluster_input_df = pd.DataFrame([{
            'title_length': len(title_input),
            'title_sentiment': title_senti,
            'has_description': int(bool(desc_input.strip())),
            'desc_sentiment': desc_senti,
            'tag_count': tags_input,
            'duration_sec': dur_sec,
            'hour': hour_input,
            'log_viewCount': log_pred
        }])
        cluster_id = kmeans_model.predict(cluster_input_df)[0]
        pop_label = get_cluster_label(kmeans_model, cluster_id)

        # Result colors
        color_map = {"Viral": "#FF3366", "High": "#FF6B35", "Medium": "#00C87A", "Low": "#4040A0"}
        box_map   = {"Viral": "result-viral", "High": "result-high", "Medium": "result-medium", "Low": "result-low"}
        emoji_map = {"Viral": "🚀", "High": "🔥", "Medium": "📈", "Low": "📉"}
        color = color_map.get(pop_label, "#FF3366")
        box   = box_map.get(pop_label, "result-low")
        emoji = emoji_map.get(pop_label, "🎥")

        col_res, col_details = st.columns([1, 1])
        with col_res:
            st.markdown(f"""
            <div class='{box}'>
                <div style='font-size:3.5rem; margin-bottom:0.5rem;'>{emoji}</div>
                <div style='font-family:"Bebas Neue",sans-serif; font-size:3rem; color:{color}; line-height:1;'>{pop_label}</div>
                <div style='font-family:"DM Mono",monospace; font-size:0.72rem; color:#5050A0; text-transform:uppercase; letter-spacing:0.1em; margin:0.5rem 0;'>Popularity Category</div>
                <div style='font-family:"Bebas Neue",sans-serif; font-size:2.2rem; color:#E8E8FF;'>{view_pred:,}</div>
                <div style='font-family:"DM Mono",monospace; font-size:0.7rem; color:#5050A0; text-transform:uppercase; letter-spacing:0.1em;'>Predicted Views</div>
            </div>
            """, unsafe_allow_html=True)

        with col_details:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Input Analysis**")
            details = [
                ("Title Length", f"{len(title_input)} chars"),
                ("Title Sentiment", f"{title_senti:+.3f}"),
                ("Desc Sentiment", f"{desc_senti:+.3f}"),
                ("Duration", f"{dur_sec:.0f}s"),
                ("Upload Hour", f"{hour_input}:00"),
                ("Tag Count", str(tags_input)),
                ("Channel", channel_input),
            ]
            for label, val in details:
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid #1E1E30;'>
                    <span style='font-family:"DM Mono",monospace; font-size:0.75rem; color:#5050A0;'>{label}</span>
                    <span style='font-family:"DM Mono",monospace; font-size:0.75rem; color:#C0C0E0;'>{val}</span>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            max_log = np.log1p(10_000_000)
            pct = min(100, int(log_pred / max_log * 100))
            st.markdown(f"""
            <div style='margin-top:1rem;'>
                <div style='font-family:"DM Mono",monospace; font-size:0.68rem; color:#5050A0; margin-bottom:0.4rem;'>RELATIVE REACH SCORE</div>
                <div style='background:#1A1A28; border-radius:999px; height:8px; overflow:hidden;'>
                    <div style='height:100%; width:{pct}%; background:linear-gradient(90deg,{color},{"#FF6B35" if color=="#FF3366" else color}); border-radius:999px;'></div>
                </div>
                <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#4040A0; margin-top:0.25rem;'>{pct}% of max scale</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════
with tab2:
    if reg_model is None:
        st.markdown("<div class='info-box'>⚠️ Dataset required to show model performance.</div>", unsafe_allow_html=True)
    else:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{model_r2:.3f}", "R² Score"),
            (c2, f"{model_rmse:.3f}", "RMSE (log scale)"),
            (c3, f"{model_sil:.3f}", "Silhouette Score"),
            (c4, "4", "Cluster Count"),
        ]:
            col.markdown(f"""
            <div class='metric-block'>
                <div class='metric-val'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Actual vs Predicted Views")
            # Sample for chart speed
            sample_df = df_trained.sample(min(500, len(df_trained)), random_state=42)
            X_sample = sample_df[REGRESSION_FEATURES]
            y_actual = sample_df['log_viewCount']
            y_pred_sample = reg_model.predict(X_sample)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_actual, y=y_pred_sample, mode='markers',
                marker=dict(color='#FF3366', size=4, opacity=0.5),
                name='Predictions'
            ))
            mn, mx = float(y_actual.min()), float(y_actual.max())
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                     line=dict(color='#4040A0', dash='dash'), name='Perfect fit'))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0F0F18',
                xaxis=dict(title='Actual (log)', color='#5050A0', gridcolor='#1E1E30'),
                yaxis=dict(title='Predicted (log)', color='#5050A0', gridcolor='#1E1E30'),
                legend=dict(font=dict(color='#C0C0E0')),
                margin=dict(t=10, b=40, l=40, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### KMeans Cluster Distribution (PCA)")
            X_cluster_df = df_trained[CLUSTER_FEATURES].dropna()
            pca_coords = PCA(n_components=2).fit_transform(X_cluster_df)
            cluster_labels = df_trained.loc[X_cluster_df.index, 'kmeans_cluster']
            label_names = [get_cluster_label(kmeans_model, c) for c in cluster_labels]

            color_seq = {'Low': '#4040A0', 'Medium': '#00C87A', 'High': '#FF6B35', 'Viral': '#FF3366'}
            fig2 = px.scatter(
                x=pca_coords[:, 0], y=pca_coords[:, 1],
                color=label_names,
                color_discrete_map=color_seq,
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                opacity=0.5
            )
            fig2.update_traces(marker_size=3)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0F0F18',
                xaxis=dict(color='#5050A0', gridcolor='#1E1E30'),
                yaxis=dict(color='#5050A0', gridcolor='#1E1E30'),
                legend=dict(font=dict(color='#C0C0E0'), bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=40, l=40, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Feature importance
        st.markdown("#### Feature Importances")
        regressor = reg_model.named_steps['regressor']
        importances = regressor.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': REGRESSION_FEATURES,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(12)

        fig3 = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'],
            orientation='h',
            marker=dict(color='#FF3366', opacity=0.8)
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0F0F18',
            xaxis=dict(title='Importance', color='#5050A0', gridcolor='#1E1E30'),
            yaxis=dict(color='#A0A0D0'),
            margin=dict(t=10, b=40, l=160, r=10),
            height=400,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════
with tab3:
    if df_trained is None:
        st.markdown("<div class='info-box'>⚠️ Dataset required.</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-block'><div class='metric-val'>{len(df_trained):,}</div><div class='metric-label'>Total Videos</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-block'><div class='metric-val'>{df_trained['channelTitle'].nunique() if 'channelTitle' in df_trained.columns else '—'}</div><div class='metric-label'>Channels</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-block'><div class='metric-val'>{len(REGRESSION_FEATURES)}</div><div class='metric-label'>Features Used</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Views by Upload Hour")
            views_by_hour = df_trained.groupby('hour')['viewCount'].sum().reset_index() if 'viewCount' in df_trained.columns else None
            if views_by_hour is not None:
                fig4 = go.Figure(go.Bar(
                    x=views_by_hour['hour'], y=views_by_hour['viewCount'],
                    marker=dict(color='#FF3366', opacity=0.8)
                ))
                fig4.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0F0F18',
                    xaxis=dict(title='Hour (24h)', color='#5050A0', gridcolor='#1E1E30'),
                    yaxis=dict(title='Total Views', color='#5050A0', gridcolor='#1E1E30'),
                    margin=dict(t=10, b=40, l=60, r=10),
                )
                st.plotly_chart(fig4, use_container_width=True)

        with col_b:
            st.markdown("#### Cluster Sizes")
            cluster_counts = df_trained['kmeans_cluster'].value_counts().reset_index()
            cluster_counts.columns = ['cluster', 'count']
            cluster_counts['label'] = cluster_counts['cluster'].apply(lambda x: get_cluster_label(kmeans_model, x))
            fig5 = go.Figure(go.Pie(
                labels=cluster_counts['label'],
                values=cluster_counts['count'],
                hole=0.5,
                marker=dict(colors=['#4040A0', '#00C87A', '#FF6B35', '#FF3366'])
            ))
            fig5.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#C0C0E0'), bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Dataset Sample")
        show_cols = [c for c in ['title', 'viewCount', 'likeCount', 'commentCount', 'duration_sec', 'hour', 'cluster_name'] if c in df_trained.columns]
        st.dataframe(df_trained[show_cols].head(20), use_container_width=True)
