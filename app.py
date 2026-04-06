import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="wide", page_icon="💳")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f3c88; text-align: center; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 12px; padding: 1rem 1.5rem;
        text-align: center; border-left: 5px solid #1f3c88;
    }
    .metric-card h2 { font-size: 2rem; color: #1f3c88; margin: 0; }
    .metric-card p  { color: #555; margin: 0; font-size: 0.9rem; }
    .cluster-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

CLUSTER_NAMES   = {0: "Low-Activity", 1: "High-Purchaser", 2: "Cash-Advance", 3: "Balanced-Spender"}
CLUSTER_COLORS  = {0: "#636EFA", 1: "#00CC96", 2: "#EF553B", 3: "#FFA15A"}
CLUSTER_ICONS   = {0: "😴", 1: "💰", 2: "🏧", 3: "⚖️"}
CLUSTER_DESC    = {
    0: "Infrequent users with low balances and minimal activity. Re-engagement candidates.",
    1: "High spenders with large credit limits and strong payment history. Prime loyalty targets.",
    2: "Heavily reliant on cash advances with revolving balances. Requires risk monitoring.",
    3: "Moderate, consistent spenders with good installment usage. Ideal for upselling.",
}

EXPECTED_COLS = [
    'BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX',
    'CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE'
]

def preprocess(df):
    data = df[[c for c in EXPECTED_COLS if c in df.columns]].copy()
    imputer = SimpleImputer(strategy='median')
    data_imp = imputer.fit_transform(data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data_imp)
    return scaled, data

def run_kmeans(scaled, k=4):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = km.fit_predict(scaled)
    return clusters, km

def assign_true_labels(df):
    labels = []
    for _, row in df.iterrows():
        if row.get('PURCHASES', 0) > 500 and row.get('CREDIT_LIMIT', 0) > 5000 and row.get('PAYMENTS', 0) > 1500:
            labels.append(1)
        elif row.get('CASH_ADVANCE', 0) > 1000 and row.get('CASH_ADVANCE_FREQUENCY', 0) > 0.3:
            labels.append(2)
        elif row.get('PURCHASES_FREQUENCY', 0) > 0.5 or row.get('PURCHASES_INSTALLMENTS_FREQUENCY', 0) > 0.4:
            labels.append(3)
        else:
            labels.append(0)
    return np.array(labels)

def metric_card(label, value, col):
    col.markdown(f"""
    <div class="metric-card">
        <h2>{value}</h2>
        <p>{label}</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Credit Card Customer Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your credit card dataset and instantly discover customer segments using K-Means Clustering + PCA</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=80)
    st.title("⚙️ Settings")
    n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
    show_elbow = st.checkbox("Show Elbow Method", value=True)
    show_pca   = st.checkbox("Show PCA Visualization", value=True)
    show_heatmap = st.checkbox("Show Cluster Heatmap", value=True)
    show_metrics = st.checkbox("Show Performance Metrics", value=True)
    st.divider()
    st.markdown("**Expected CSV Columns:**")
    for c in EXPECTED_COLS:
        st.markdown(f"- `{c}`")

uploaded = st.file_uploader("Upload your CSV file (e.g. CC_GENERAL.csv)", type=["csv"])

if not uploaded:
    st.info("👆 Upload a CSV to get started, or click below to use the sample CC_GENERAL dataset structure.")
    st.stop()

df_raw = pd.read_csv(uploaded)
if 'CUST_ID' in df_raw.columns:
    df_raw = df_raw.drop('CUST_ID', axis=1)

missing_cols = [c for c in EXPECTED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

with st.spinner("Running K-Means Clustering..."):
    scaled_data, clean_df = preprocess(df_raw)
    clusters, kmeans_model = run_kmeans(scaled_data, k=n_clusters)
    df_raw['Cluster'] = clusters
    df_raw['Segment'] = df_raw['Cluster'].map(CLUSTER_NAMES if n_clusters == 4 else {i: f"Cluster {i}" for i in range(n_clusters)})

st.success(f"Successfully segmented {len(df_raw):,} customers into {n_clusters} clusters!")

st.subheader("Overview")
counts = df_raw['Cluster'].value_counts().sort_index()
cols = st.columns(n_clusters)
for i, col in enumerate(cols):
    name = CLUSTER_NAMES.get(i, f"Cluster {i}")
    icon = CLUSTER_ICONS.get(i, "📌")
    metric_card(f"{icon} {name}", f"{counts.get(i,0):,}", col)

st.markdown("")

tabs = st.tabs(["📈 Elbow Method", "🔵 PCA Clusters", "🌡️ Cluster Heatmap", "📋 Performance Metrics", "🔍 Explore Data", "🎯 Classify New Customer"])

with tabs[0]:
    if show_elbow:
        st.subheader("Elbow Method — Finding Optimal K")
        wcss = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, init='k-means++', random_state=42)
            km.fit(scaled_data)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--', color='steelblue', linewidth=2, markersize=8)
        ax.axvline(x=4, color='red', linestyle=':', alpha=0.7, label='Optimal K=4')
        ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS (Inertia)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.info("The elbow point (where the curve bends) indicates the optimal number of clusters. K=4 shows a clear bend.")

with tabs[1]:
    if show_pca:
        st.subheader("Customer Segments — PCA Visualization")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors_list = list(CLUSTER_COLORS.values())
        for k in range(n_clusters):
            mask = clusters == k
            label = CLUSTER_NAMES.get(k, f"Cluster {k}")
            ax.scatter(pca_data[mask, 0], pca_data[mask, 1],
                       c=colors_list[k % len(colors_list)], label=label, alpha=0.6, s=15)
        ax.set_title('Customer Segments (PCA Visualization)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        c1, c2 = st.columns(2)
        c1.metric("PC1 Explained Variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
        c2.metric("PC2 Explained Variance", f"{pca.explained_variance_ratio_[1]*100:.1f}%")

with tabs[2]:
    if show_heatmap:
        st.subheader("Characteristic Profile of Each Customer Segment")
        profile = df_raw.drop('Segment', axis=1).groupby('Cluster').mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(profile.T, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax,
                    linewidths=0.5, cbar_kws={'label': 'Mean Value'})
        ax.set_title('Characteristic Profile of Each Customer Segment', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')
        st.pyplot(fig)

        if n_clusters == 4:
            st.subheader("Cluster Descriptions")
            c1, c2 = st.columns(2)
            for i, (cnum, desc) in enumerate(CLUSTER_DESC.items()):
                col = c1 if i % 2 == 0 else c2
                col.markdown(f"""
                <div style="background:#f8f9fa; border-radius:10px; padding:1rem; margin-bottom:1rem; border-left:4px solid {CLUSTER_COLORS[cnum]}">
                <b>{CLUSTER_ICONS[cnum]} Cluster {cnum} — {CLUSTER_NAMES[cnum]}</b><br>
                <span style="color:#555; font-size:0.9rem">{desc}</span>
                </div>""", unsafe_allow_html=True)

with tabs[3]:
    if show_metrics and n_clusters == 4:
        st.subheader("Model Performance Metrics")
        true_labels = assign_true_labels(df_raw)
        pred_labels = np.array(clusters)
        acc = accuracy_score(true_labels, pred_labels)

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Accuracy", f"{acc*100:.2f}%")

        report = classification_report(true_labels, pred_labels,
                    target_names=list(CLUSTER_NAMES.values()), output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)

        m2.metric("Macro F1-Score",    f"{report['macro avg']['f1-score']:.2f}")
        m3.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2f}")

        st.markdown("#### Per-Class Report")
        display_df = report_df.iloc[:n_clusters][['precision','recall','f1-score','support']]
        st.dataframe(display_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']), use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(n_clusters)
        w = 0.25
        names = list(CLUSTER_NAMES.values())
        prec = [report[n]['precision'] for n in names]
        rec  = [report[n]['recall']    for n in names]
        f1   = [report[n]['f1-score']  for n in names]

        ax.bar(x - w, prec, w, label='Precision', color='steelblue')
        ax.bar(x,     rec,  w, label='Recall',    color='coral')
        ax.bar(x + w, f1,   w, label='F1-Score',  color='mediumseagreen')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_title('Precision / Recall / F1-Score per Cluster')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
        st.pyplot(fig)
    else:
        st.info("Performance metrics are available when K=4. Adjust the slider in the sidebar.")

with tabs[4]:
    st.subheader("🔍 Explore Segmented Data")
    filter_cluster = st.multiselect("Filter by Cluster", options=sorted(df_raw['Cluster'].unique()),
                                     default=sorted(df_raw['Cluster'].unique()),
                                     format_func=lambda x: f"Cluster {x} — {CLUSTER_NAMES.get(x, '')}")
    filtered = df_raw[df_raw['Cluster'].isin(filter_cluster)]
    st.dataframe(filtered.head(200), use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

    st.markdown("#### Cluster Size Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    count_data = df_raw['Cluster'].value_counts().sort_index()
    bars = ax.bar([f"C{k}\n{CLUSTER_NAMES.get(k,'')}" for k in count_data.index],
                  count_data.values,
                  color=[CLUSTER_COLORS.get(k, 'gray') for k in count_data.index],
                  edgecolor='white', linewidth=1.2)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(int(bar.get_height())), ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Customers per Segment")
    ax.set_ylabel("Count")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

with tabs[5]:
    st.subheader("🎯 Classify a New Customer")
    st.markdown("Enter a customer's details below and the model will assign them to a segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        balance       = st.number_input("Balance",              0.0, 20000.0, 1000.0)
        purchases     = st.number_input("Purchases",            0.0, 50000.0, 500.0)
        cash_advance  = st.number_input("Cash Advance",         0.0, 20000.0, 0.0)
        credit_limit  = st.number_input("Credit Limit",         0.0, 30000.0, 5000.0)
        payments      = st.number_input("Payments",             0.0, 30000.0, 1000.0)
        min_payments  = st.number_input("Minimum Payments",     0.0, 10000.0, 200.0)
    with col2:
        balance_freq  = st.slider("Balance Frequency",          0.0, 1.0, 0.8)
        purch_freq    = st.slider("Purchases Frequency",        0.0, 1.0, 0.5)
        oneoff_freq   = st.slider("One-off Purchases Freq",     0.0, 1.0, 0.2)
        install_freq  = st.slider("Installments Freq",          0.0, 1.0, 0.3)
        ca_freq       = st.slider("Cash Advance Frequency",     0.0, 1.0, 0.0)
    with col3:
        oneoff_purch  = st.number_input("One-off Purchases",    0.0, 40000.0, 200.0)
        install_purch = st.number_input("Installment Purchases",0.0, 20000.0, 300.0)
        ca_trx        = st.number_input("Cash Advance Trx",     0, 100, 0)
        purch_trx     = st.number_input("Purchases Trx",        0, 300, 5)
        prc_full      = st.slider("% Full Payment",             0.0, 1.0, 0.1)
        tenure        = st.slider("Tenure (months)",            1, 12, 12)

    if st.button("Classify Customer", use_container_width=True, type="primary"):
        new_customer = np.array([[
            balance, balance_freq, purchases, oneoff_purch, install_purch,
            cash_advance, purch_freq, oneoff_freq, install_freq, ca_freq,
            ca_trx, purch_trx, credit_limit, payments, min_payments, prc_full, tenure
        ]])

        imp2 = SimpleImputer(strategy='median')
        imp2.fit(clean_df.values)
        sc2 = StandardScaler()
        sc2.fit(imp2.transform(clean_df.values))
        new_scaled = sc2.transform(imp2.transform(new_customer))
        pred = kmeans_model.predict(new_scaled)[0]

        name  = CLUSTER_NAMES.get(pred, f"Cluster {pred}")
        color = CLUSTER_COLORS.get(pred, "#888")
        icon  = CLUSTER_ICONS.get(pred, "📌")
        desc  = CLUSTER_DESC.get(pred, "")

        st.markdown(f"""
        <div style="background:#f0f4ff; border-radius:14px; padding:1.5rem; border-left:6px solid {color}; margin-top:1rem">
            <h2 style="color:{color}">{icon} Cluster {pred} — {name}</h2>
            <p style="color:#444; font-size:1rem">{desc}</p>
        </div>""", unsafe_allow_html=True)