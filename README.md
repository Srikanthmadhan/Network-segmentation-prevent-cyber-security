
# 🔒 AI-Driven Network Segmentation & Behavioral Analysis

A Streamlit-based real-time network monitoring dashboard that learns baseline behavior from live traffic, automatically segments flows using unsupervised learning, and detects anomalies with Isolation Forest. [file:1]

---

## Features

- **Baseline learning**: Captures live network connections over a configurable time window to build a behavioral baseline of flows, ports, and protocols. [file:1]  
- **Behavioral segmentation**: Clusters flows into segments using K-Means on engineered features such as port statistics, connection counts, and duration. [file:1]  
- **Auto segment discovery**: Optional elbow-method–based auto-detection of the optimal number of clusters. [file:1]  
- **Anomaly detection**: Uses Isolation Forest on baseline features to flag anomalous live flows with scores and labels. [file:1]  
- **Policy inference**: Learns trusted flows and segments from baseline behavior and classifies live flows as Trusted, New Flow, Segment Drift, or Anomalous. [file:1]  
- **Real-time dashboard**: Streamlit UI with metrics, tables, and Plotly visualizations for segments, anomalies, and PCA-based 3D embedding. [file:1]  
- **Security insights**: High-level findings and recommended actions based on current anomaly and risk profile. [file:1]  

---

## Tech Stack

- **Language**: Python 3.x [file:1]  
- **Framework**: Streamlit [file:1]  
- **System / Networking**: `psutil` for live network connections [file:1]  
- **Data & ML**:
  - `pandas`, `numpy` for data handling [file:1]
  - `scikit-learn` (KMeans, IsolationForest, StandardScaler, PCA) for clustering, anomaly detection, scaling, and visualization [file:1]  
- **Visualization**: Plotly (Express & Graph Objects) for charts and 3D plots [file:1]  

---

## Installation

1. **Clone the repository**

   ```
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment** (recommended)

   ```
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   Create a `requirements.txt` such as:

   ```
   streamlit
   psutil
   pandas
   numpy
   plotly
   scikit-learn
   ```

   Then install:

   ```
   pip install -r requirements.txt
   ```

4. **OS / permissions**

   - Run on a machine where `psutil.net_connections(kind="inet")` can access active connections. [file:1]  
   - On some systems you may need administrator/root privileges for full visibility. [file:1]  

---

## Usage

1. **Start the Streamlit app**

   ```
   streamlit run app.py
   ```

2. **Open the UI**

   - Streamlit will print a local URL (e.g. `http://localhost:8501`) in the terminal.  
   - Open it in a browser on the same machine.

3. **Configure in the sidebar**

   In the left sidebar: [file:1]

   - **Baseline Window (seconds)**: Duration to observe traffic and learn baseline (default 30s). [file:1]  
   - **Target Segments**: Number of behavioral clusters for K-Means (default 4). [file:1]  
   - **Auto-detect optimal segments**: Let the app run the elbow method and override `Target Segments`. [file:1]  
   - **Anomaly Sensitivity**: Isolation Forest contamination (lower → stricter). [file:1]  
   - **Live Refresh Interval (seconds)**: Controls the cadence of live snapshots. [file:1]  
   - **Show Advanced Analytics**: Toggles PCA and anomaly score plots. [file:1]  
   - **Enable Auto-Refresh**: Optional automatic rerun for continuous updates. [file:1]  

4. **Workflow inside the app**

   - **Phase 1 – Baseline Traffic Learning**  
     - The app captures snapshots of `psutil.net_connections(kind="inet")` until the baseline timer completes. [file:1]  
     - If no traffic is seen, troubleshooting hints are shown and execution stops. [file:1]  

   - **Phase 2 – Feature Engineering & Segmentation**  
     - Flows are identified as `src_ip → dst_ip`. [file:1]  
     - Aggregated features per flow: port stats, connection counts, unique ports, duration, protocol. [file:1]  
     - Features are standardized with `StandardScaler`, then clustered by K-Means. [file:1]  
     - Segment distribution and profiles are shown via pie chart and table. [file:1]  

   - **Phase 3 – Policy Inference**  
     - Builds an in-memory baseline of flows, ports, and protocols. [file:1]  
     - Displays metrics: learned segments, unique flows, active IPs. [file:1]  

   - **Phase 4 – Live Traffic Monitoring**  
     - Captures a new live snapshot, recomputes features, and assigns each live flow to a segment using the trained scaler and K-Means model. [file:1]  

   - **Anomaly Detection & Policy Evaluation**  
     - Trains Isolation Forest on baseline features using the configured contamination rate. [file:1]  
     - Scores live flows, labels them as NORMAL / ANOMALOUS, and classifies flows into Trusted, New Flow, Segment Drift, or Anomalous decisions. [file:1]  
     - Computes a simple risk score combining anomaly, novelty, and segment drift flags. [file:1]  

   - **Real-Time Security Dashboard**  
     - Shows metrics: Active Flows, Anomalies (+%), New Flows, High Risk flows. [file:1]  
     - Displays a sortable dataframe with flow id, segment, protocol, connection count, anomaly label, decision, and risk score. [file:1]  

   - **Advanced Analytics (optional)**  
     - 3D PCA view of baseline vs live flows colored by segment, with live flows highlighted. [file:1]  
     - Histogram of anomaly scores with an (approximate) threshold line derived from baseline scores. [file:1]  

   - **Security Insights**  
     - Key findings (counts of anomalies, new flows, high-risk flows, most active segment). [file:1]  
     - Recommendations for investigation, allow-listing, and tuning sensitivity/segment count. [file:1]  

---

## Notes & Limitations

- The app **does not** block or modify traffic; it is a monitoring and analysis tool only. [file:1]  
- Visibility depends on OS and privileges; limited permissions may reduce connection coverage. [file:1]  
- Baseline quality depends on the representativeness of traffic during the baseline window. [file:1]  
- This is designed primarily for single-host visibility (connections as seen from the running machine). [file:1]  

---

## Project Structure

```
.
├── app.py          # Streamlit application with all logic (UI, ML, analytics)
└── README.md      #you're here
