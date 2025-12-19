import streamlit as st
import psutil
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI-Driven Network Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔒 AI-Driven Network Segmentation & Behavioral Analysis")
st.markdown("*Autonomous network discovery, segmentation, and anomaly detection*")

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Learning Parameters")
    BASELINE_SECONDS = st.slider(
        "Baseline Window (seconds)", 
        min_value=10, 
        max_value=180, 
        value=30,
        help="Duration to observe traffic and establish behavioral baseline"
    )
    
    st.subheader("Segmentation")
    N_SEGMENTS = st.slider(
        "Target Segments", 
        min_value=2, 
        max_value=10, 
        value=4,
        help="Number of behavioral clusters to identify"
    )
    
    AUTO_SEGMENTS = st.checkbox(
        "Auto-detect optimal segments",
        value=False,
        help="Use elbow method to find best cluster count"
    )
    
    st.subheader("Anomaly Detection")
    ANOMALY_RATE = st.slider(
        "Anomaly Sensitivity", 
        min_value=0.01, 
        max_value=0.3, 
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalous traffic (lower = stricter)"
    )
    
    st.subheader("Monitoring")
    LIVE_REFRESH = st.slider(
        "Live Refresh Interval (seconds)",
        min_value=1,
        max_value=30,
        value=5
    )
    
    SHOW_ADVANCED = st.checkbox("Show Advanced Analytics", value=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_connection_type(port):
    """Classify connection by port"""
    port_map = {
        range(0, 1024): "System/Well-Known",
        range(1024, 49152): "Registered",
        range(49152, 65536): "Dynamic/Private"
    }
    try:
        if port is None:
            return "Unknown"
        for port_range, name in port_map.items():
            if port in port_range:
                return name
    except TypeError:
        return "Unknown"
    return "Unknown"

def classify_protocol(dst_port):
    """Identify common protocols"""
    protocols = {
        80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP",
        25: "SMTP", 53: "DNS", 110: "POP3", 143: "IMAP",
        3306: "MySQL", 5432: "PostgreSQL", 6379: "Redis",
        27017: "MongoDB", 3389: "RDP", 5900: "VNC"
    }
    if dst_port is None:
        return "Unknown"
    return protocols.get(dst_port, f"Port-{dst_port}")

@st.cache_data(ttl=2)
def capture_connections():
    """Capture active network connections with enhanced metadata"""
    rows = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.raddr:
                rows.append({
                    "src_ip": c.laddr.ip,
                    "dst_ip": c.raddr.ip,
                    "src_port": c.laddr.port,
                    "dst_port": c.raddr.port,
                    "status": c.status,
                    "protocol": classify_protocol(c.raddr.port),
                    "port_category": get_connection_type(c.raddr.port),
                    "timestamp": time.time()
                })
    except (psutil.AccessDenied, PermissionError):
        st.error("⚠️ Permission denied. Run with elevated privileges for full network visibility.")
    except Exception as e:
        st.error(f"Error capturing connections: {str(e)}")
    
    return pd.DataFrame(rows)

def find_optimal_clusters(data, max_k=10):
    """Use elbow method to find optimal cluster count"""
    inertias = []
    # ensure we have at least a few samples
    n = max(0, len(data))
    if n < 3:
        return 3, range(2, min(max_k + 1, 3)), []

    K_range = range(2, min(max_k + 1, n))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Calculate elbow using rate of change
    if len(inertias) > 2:
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        optimal_k = int(np.argmax(second_deltas) + 2)
        return optimal_k, K_range, inertias

    return 3, K_range, inertias

# ============================================================================
# BASELINE LEARNING PHASE
# ============================================================================

st.header("📊 Phase 1: Baseline Traffic Learning")

progress_bar = st.progress(0)
status_text = st.empty()
baseline_data = []

start_time = time.time()

try:
    while time.time() - start_time < BASELINE_SECONDS:
        elapsed = time.time() - start_time
        progress = elapsed / BASELINE_SECONDS
        progress_bar.progress(min(progress, 1.0))
        
        df = capture_connections()
        if not df.empty:
            baseline_data.append(df)
            status_text.info(f"⏳ Learning... {len(baseline_data)} snapshots captured ({int(elapsed)}s / {BASELINE_SECONDS}s)")
        else:
            status_text.warning(f"⏳ Waiting for traffic... ({int(elapsed)}s / {BASELINE_SECONDS}s)")
        
        time.sleep(1)
    
    progress_bar.progress(1.0)
    
except KeyboardInterrupt:
    st.warning("Baseline learning interrupted by user")

if not baseline_data:
    st.error("❌ No network traffic observed during baseline period. Cannot proceed.")
    st.info("💡 **Troubleshooting:**\n- Ensure there's active network traffic\n- Run with administrator/root privileges\n- Try increasing the baseline window")
    st.stop()

baseline_df = pd.concat(baseline_data, ignore_index=True)
status_text.success(f"✅ Baseline complete: {len(baseline_df)} connections captured from {baseline_df['src_ip'].nunique()} sources to {baseline_df['dst_ip'].nunique()} destinations")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

st.header("🔬 Phase 2: Feature Engineering & Segmentation")

# Create flow identifiers
baseline_df["flow_id"] = baseline_df["src_ip"] + " → " + baseline_df["dst_ip"]
baseline_df["reverse_flow_id"] = baseline_df["dst_ip"] + " → " + baseline_df["src_ip"]

# Aggregate flow statistics
flow_stats = baseline_df.groupby("flow_id").agg(
    src_ip=("src_ip", "first"),
    dst_ip=("dst_ip", "first"),
    src_port_mean=("src_port", "mean"),
    src_port_std=("src_port", "std"),
    dst_port_mean=("dst_port", "mean"),
    dst_port_std=("dst_port", "std"),
    connection_count=("timestamp", "count"),
    unique_src_ports=("src_port", "nunique"),
    unique_dst_ports=("dst_port", "nunique"),
    duration=("timestamp", lambda x: x.max() - x.min()),
    protocol=("protocol", lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
).reset_index()

flow_stats.fillna(0, inplace=True)

# Feature scaling
feature_cols = ["src_port_mean", "dst_port_mean", "connection_count", 
                "unique_src_ports", "unique_dst_ports", "duration"]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(flow_stats[feature_cols])

# ============================================================================
# CLUSTERING / SEGMENTATION
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Network Segmentation")
    
    if AUTO_SEGMENTS and len(flow_stats) >= 4:
        optimal_k, k_range, inertias = find_optimal_clusters(scaled_features, max_k=10)
        st.info(f"🤖 Auto-detected optimal segments: **{optimal_k}**")
        N_SEGMENTS = optimal_k
        
        # Show elbow plot
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), 
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red", 
                           annotation_text=f"Optimal: {optimal_k}")
        fig_elbow.update_layout(
            title="Elbow Method Analysis",
            xaxis_title="Number of Clusters",
            yaxis_title="Inertia",
            height=300
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

# Perform clustering
kmeans = KMeans(n_clusters=N_SEGMENTS, random_state=42, n_init=10)
flow_stats["segment"] = kmeans.fit_predict(scaled_features)

# Create segment profiles
segment_profiles = flow_stats.groupby("segment").agg({
    "connection_count": ["mean", "sum"],
    "dst_port_mean": "mean",
    "protocol": lambda x: x.mode()[0] if len(x) > 0 else "Mixed",
    "flow_id": "count"
}).round(2)

segment_profiles.columns = ["Avg Connections", "Total Connections", "Avg Port", "Primary Protocol", "Flow Count"]

with col2:
    st.subheader("📈 Segment Distribution")
    
    segment_counts = flow_stats["segment"].value_counts().sort_index()
    fig_dist = px.pie(
        values=segment_counts.values,
        names=[f"Segment {i}" for i in segment_counts.index],
        title="Flow Distribution by Segment",
        hole=0.4
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("🔍 Segment Profiles")
st.dataframe(segment_profiles, use_container_width=True)

# ============================================================================
# POLICY INFERENCE
# ============================================================================

st.header("📋 Phase 3: Policy Inference")

# Build allowed segment pairs and flow behaviors
allowed_segments = set(flow_stats["segment"].unique())
flow_to_segment = dict(zip(flow_stats["flow_id"], flow_stats["segment"]))

baseline_behavior = {}
for _, row in baseline_df.iterrows():
    flow = f"{row['src_ip']} → {row['dst_ip']}"
    if flow not in baseline_behavior:
        baseline_behavior[flow] = {
            "ports": set(),
            "protocols": set(),
            "count": 0
        }
    baseline_behavior[flow]["ports"].add(row["dst_port"])
    baseline_behavior[flow]["protocols"].add(row["protocol"])
    baseline_behavior[flow]["count"] += 1

col1, col2, col3 = st.columns(3)
col1.metric("📊 Learned Segments", len(allowed_segments))
col2.metric("🔄 Unique Flows", len(baseline_behavior))
col3.metric("🌐 Active IPs", baseline_df["src_ip"].nunique() + baseline_df["dst_ip"].nunique())

# ============================================================================
# LIVE MONITORING
# ============================================================================

st.header("🔴 Phase 4: Live Traffic Monitoring")

# Capture live traffic
live_df = capture_connections()

if live_df.empty:
    st.warning("⚠️ No live traffic detected. Waiting for connections...")
    st.stop()

live_df["flow_id"] = live_df["src_ip"] + " → " + live_df["dst_ip"]

# Aggregate live flow features
live_flow_stats = live_df.groupby("flow_id").agg(
    src_ip=("src_ip", "first"),
    dst_ip=("dst_ip", "first"),
    src_port_mean=("src_port", "mean"),
    src_port_std=("src_port", "std"),
    dst_port_mean=("dst_port", "mean"),
    dst_port_std=("dst_port", "std"),
    connection_count=("timestamp", "count"),
    unique_src_ports=("src_port", "nunique"),
    unique_dst_ports=("dst_port", "nunique"),
    duration=("timestamp", lambda x: x.max() - x.min()),
    protocol=("protocol", lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
).reset_index()

live_flow_stats.fillna(0, inplace=True)

# Scale and predict segments
live_scaled = scaler.transform(live_flow_stats[feature_cols])
live_flow_stats["segment"] = kmeans.predict(live_scaled)

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

st.subheader("🚨 Anomaly Detection")

# Train isolation forest on baseline
iso_forest = IsolationForest(
    contamination=ANOMALY_RATE, 
    random_state=42,
    n_estimators=100
)
iso_forest.fit(scaled_features)

# Predict anomalies in live traffic
live_flow_stats["anomaly_score"] = iso_forest.score_samples(live_scaled)
live_flow_stats["is_anomaly"] = iso_forest.predict(live_scaled)
live_flow_stats["anomaly_label"] = live_flow_stats["is_anomaly"].map({-1: "⚠️ ANOMALOUS", 1: "✅ NORMAL"})

# ============================================================================
# POLICY EVALUATION
# ============================================================================

def evaluate_flow(row):
    """Comprehensive flow evaluation"""
    flow_id = row["flow_id"]
    segment = row["segment"]
    is_anomaly = row["is_anomaly"] == -1
    
    # Check if flow was in baseline
    known_flow = flow_id in baseline_behavior
    known_segment = segment in allowed_segments
    
    if is_anomaly:
        return "🔴 ANOMALOUS"
    elif not known_flow:
        return "🟡 NEW FLOW"
    elif not known_segment:
        return "🟠 SEGMENT DRIFT"
    else:
        return "🟢 TRUSTED"

live_flow_stats["decision"] = live_flow_stats.apply(evaluate_flow, axis=1)

# Calculate risk scores
live_flow_stats["risk_score"] = (
    (live_flow_stats["is_anomaly"] == -1).astype(int) * 50 +
    (~live_flow_stats["flow_id"].isin(baseline_behavior.keys())).astype(int) * 30 +
    (~live_flow_stats["segment"].isin(allowed_segments)).astype(int) * 20
)

# ============================================================================
# DASHBOARD
# ============================================================================

st.subheader("📊 Real-Time Security Dashboard")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

total_flows = len(live_flow_stats)
anomalous = (live_flow_stats["is_anomaly"] == -1).sum()
new_flows = (~live_flow_stats["flow_id"].isin(baseline_behavior.keys())).sum()
high_risk = (live_flow_stats["risk_score"] >= 50).sum()

metric_col1.metric("🌐 Active Flows", total_flows)
# avoid division by zero in delta
try:
    delta_pct = f"{(anomalous/total_flows*100):.1f}%"
except Exception:
    delta_pct = "0%"
metric_col2.metric("🚨 Anomalies", anomalous, delta=delta_pct, delta_color="inverse")
metric_col3.metric("🆕 New Flows", new_flows)
metric_col4.metric("⚠️ High Risk", high_risk)

# Flow table with color coding
st.subheader("🔍 Flow Analysis")

display_df = live_flow_stats[[
    "flow_id", "segment", "protocol", "connection_count", 
    "anomaly_label", "decision", "risk_score"
]].sort_values("risk_score", ascending=False)

# Some Streamlit versions may not support column_config.ProgressColumn —
# show a plain dataframe for compatibility.
st.dataframe(display_df, use_container_width=True)

# ============================================================================
# ADVANCED ANALYTICS
# ============================================================================

if SHOW_ADVANCED:
    st.header("🔬 Advanced Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Segment Distribution (3D PCA)")
        
        # PCA for visualization
        if len(scaled_features) >= 3:
            pca = PCA(n_components=3)
            pca_baseline = pca.fit_transform(scaled_features)
            pca_live = pca.transform(live_scaled)
            
            fig_3d = go.Figure()
            
            # Baseline flows
            fig_3d.add_trace(go.Scatter3d(
                x=pca_baseline[:, 0],
                y=pca_baseline[:, 1],
                z=pca_baseline[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=flow_stats["segment"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Segment"),
                    opacity=0.6
                ),
                name='Baseline',
                text=flow_stats["flow_id"]
            ))
            
            # Live flows
            fig_3d.add_trace(go.Scatter3d(
                x=pca_live[:, 0],
                y=pca_live[:, 1],
                z=pca_live[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                name='Live Traffic',
                text=live_flow_stats["flow_id"]
            ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                height=500
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
    
    with viz_col2:
        st.subheader("Anomaly Score Distribution")
        
        fig_anomaly = go.Figure()
        
        fig_anomaly.add_trace(go.Histogram(
            x=live_flow_stats["anomaly_score"],
            nbinsx=30,
            name='Distribution',
            marker_color='steelblue'
        ))
        
        # Add threshold line - compute from baseline scores to avoid relying on private attributes
        try:
            baseline_scores = iso_forest.score_samples(scaled_features)
            threshold = np.percentile(baseline_scores, ANOMALY_RATE * 100)
            fig_anomaly.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:.3f}"
            )
        except Exception:
            pass
        
        fig_anomaly.update_layout(
            xaxis_title="Anomaly Score (lower = more anomalous)",
            yaxis_title="Count",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True)

# ============================================================================
# INSIGHTS & RECOMMENDATIONS
# ============================================================================

st.header("💡 Security Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.subheader("🎯 Key Findings")
    
    if anomalous > 0:
        st.warning(f"⚠️ **{anomalous} anomalous flows detected** - Review for potential threats")
    
    if new_flows > total_flows * 0.3:
        st.info(f"ℹ️ **{(new_flows/total_flows*100):.0f}% new flows** - Network behavior is evolving")
    
    if high_risk > 0:
        st.error(f"🚨 **{high_risk} high-risk flows** - Immediate investigation recommended")
    else:
        st.success("✅ **No high-risk flows detected** - Network behavior is within expected parameters")
    
    # Most active segment (guard against empty)
    try:
        most_active_seg = int(live_flow_stats["segment"].mode()[0])
    except Exception:
        most_active_seg = "N/A"
    st.info(f"📊 **Most active segment: {most_active_seg}** - Represents primary traffic pattern")

with insights_col2:
    st.subheader("📝 Recommendations")
    
    st.markdown("""
    **Immediate Actions:**
    - 🔍 Investigate all flows with risk scores ≥ 70
    - 🔒 Consider isolating anomalous flows for deeper inspection
    - 📊 Monitor new flows for pattern establishment
    
    **Policy Considerations:**
    - ✅ Trusted flows can be allow-listed
    - ⚠️ Anomalous flows require manual review
    - 🆕 New flows need time to establish behavioral baseline
    
    **System Health:**
    - 🔄 Baseline should be refreshed periodically
    - 📈 Segment count may need adjustment if traffic patterns change
    - 🎯 Anomaly sensitivity can be tuned based on false positive rate
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption("""
**🤖 Zero-Trust Network Segmentation Philosophy:**
- 📊 Segments are **dynamically learned** from traffic behavior, not predefined roles
- 🎯 Policies are **inferred from observed patterns**, not static rules
- 🚨 Anomalies are **flagged for investigation**, not automatically blocked
- 🔒 Trust is **continuously evaluated** based on behavioral consistency
- 🌐 **No hardcoded assumptions** about network topology or application roles

*Built with Streamlit, scikit-learn, and Plotly | Real-time behavioral analytics*
""")

# Auto-refresh for live monitoring
if st.sidebar.checkbox("Enable Auto-Refresh", value=False):
    st.rerun()
