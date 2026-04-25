import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("nhs_clean_final.csv")

data = load_data()

st.title("NHS COVID Structural Impact Dashboard")

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.header("Global Filters")

year_range = st.sidebar.slider(
    "Year Range",
    int(data.year.min()),
    int(data.year.max()),
    (2019, 2024)
)

phase_filter = st.sidebar.multiselect(
    "COVID Phase",
    options=data.covid_phase.unique(),
    default=list(data.covid_phase.unique())
)

top_n = st.sidebar.slider("Top Diagnoses (by admissions)", 5, 50, 20)

# Apply filters
df = data[
    (data.year >= year_range[0]) &
    (data.year <= year_range[1]) &
    (data.covid_phase.isin(phase_filter))
]

# ---------------------------
# TOP-N FILTER
# ---------------------------
top_diag = (
    df.groupby("diagnosis_group")["admissions"]
    .sum()
    .nlargest(top_n)
    .index
)

df = df[df.diagnosis_group.isin(top_diag)]

# ---------------------------
# 🔴 ANOMALY DETECTION
# ---------------------------
df['z_score'] = (
    df['hospital_pressure_index'] - df['hospital_pressure_index'].mean()
) / df['hospital_pressure_index'].std()

df['is_anomaly'] = df['z_score'].abs() > 2

# ---------------------------
# USER DIAGNOSIS SELECTION (CROSS-FILTER)
# ---------------------------
selected_diag = st.sidebar.multiselect(
    "Highlight Diagnosis",
    options=sorted(df.diagnosis_group.unique())
)

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌟 System Dynamics",
    "🔁 Admission Flow",
    "🌳 Diagnosis Burden",
    "📊 Distribution & Anomalies",
    "📘 Derived Metrics"
])

# =========================================================
# 🌟 HERO TAB
# =========================================================
with tab1:

    st.header("System-Level Structural Dynamics")

    features = [
        'emergency_rate',
        'elderly_ratio',
        'mean_los',
        'mean_wait',
        'hospital_pressure_index'
    ]

    selected_features = st.multiselect(
        "Select Features",
        features,
        default=features
    )

    color_feature = st.selectbox(
        "Color By",
        ['year', 'hospital_pressure_index', 'elderly_ratio']
    )

    # ---------------------------
    # 🔴 FIX: SCALE DATA FOR VISIBILITY
    # ---------------------------
    plot_df = df.copy()

    # Normalize each feature (critical for readability)
    for col in selected_features:
        if col in plot_df.columns:
            min_val = plot_df[col].min()
            max_val = plot_df[col].max()
            if max_val - min_val != 0:
                plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)

    fig = px.parallel_coordinates(
        plot_df,
        dimensions=selected_features,
        color=color_feature,
        color_continuous_scale=px.colors.diverging.Tealrose
    )

    # ---------------------------
    # 🔴 FIX: IMPROVE LAYOUT
    # ---------------------------
    fig.update_layout(
        height=750,  # bigger height
        margin=dict(l=80, r=80, t=60, b=60),
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    Note: Features are normalized (0–1) for comparability across dimensions.
    """)

# =========================================================
# 🔁 SANKEY TAB
# =========================================================
with tab2:

    st.header("Admission Flow Transformation")

    df_sankey = df.groupby('covid_phase').agg({
        'emergency': 'sum',
        'admissions': 'sum'
    }).reset_index()

    labels = ['Pre-COVID', 'COVID Peak', 'Post-COVID', 'Emergency', 'Non-Emergency']

    source = [0,1,2,0,1,2]
    target = [3,3,3,4,4,4]

    values = []

    for phase in ['pre_covid','covid_peak','post_covid']:
        subset = df_sankey[df_sankey.covid_phase == phase]
        if not subset.empty:
            emergency = subset['emergency'].values[0]
            total = subset['admissions'].values[0]
        else:
            emergency, total = 0,0
        values.append(emergency)

    for phase in ['pre_covid','covid_peak','post_covid']:
        subset = df_sankey[df_sankey.covid_phase == phase]
        if not subset.empty:
            emergency = subset['emergency'].values[0]
            total = subset['admissions'].values[0]
        else:
            emergency, total = 0,0
        values.append(total - emergency)

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=20, thickness=20),
        link=dict(source=source, target=target, value=values)
    ))

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🌳 TREEMAP TAB
# =========================================================
with tab3:

    st.header("Diagnosis Burden & Aging")

    color_metric = st.selectbox(
        "Color Metric",
        ['elderly_ratio', 'hospital_pressure_index', 'emergency_rate']
    )

    # ---------------------------
    # 🔴 CLEAN + PREP DATA
    # ---------------------------
    treemap_df = df.copy()

    treemap_df = treemap_df.dropna(subset=['covid_phase', 'diagnosis_group', 'admissions'])

    treemap_df['admissions'] = pd.to_numeric(treemap_df['admissions'], errors='coerce')

    treemap_df = treemap_df[treemap_df['admissions'] > 0]

    # ---------------------------
    # 🔴 COMPUTE % OF PARENT
    # ---------------------------
    treemap_df['percent_of_phase'] = (
        treemap_df['admissions'] /
        treemap_df.groupby('covid_phase')['admissions'].transform('sum')
    ) * 100

    # ---------------------------
    # 🔴 TREEMAP
    # ---------------------------
    fig = px.treemap(
        treemap_df,
        path=['covid_phase', 'diagnosis_group'],
        values='admissions',
        color=color_metric,
        color_continuous_scale='RdBu',
        hover_data={
            'group_description': True,
            'percent_of_phase': ':.2f',
            'admissions': ':,'
        }
    )

    # ---------------------------
    # 🔴 SHOW % LABELS INSIDE BLOCKS
    # ---------------------------
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[1]:.1f}%",
        textfont_size=12
    )

    # IMPORTANT: ensure customdata includes percent
    fig.data[0].customdata = np.stack(
        (
            treemap_df['group_description'],
            treemap_df['percent_of_phase']
        ),
        axis=-1
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    Block size shows total admissions, while percentages reveal **relative dominance within each COVID phase**.  
    This exposes shifts in healthcare prioritisation — not just volume changes.
    """)

# =========================================================
# 📊 DISTRIBUTION + ANOMALY TAB
# =========================================================
with tab4:

    st.header("Distribution of System Pressure")

    fig = px.violin(
        df,
        x='covid_phase',
        y='hospital_pressure_index',
        box=True,
        points="all",
        color='covid_phase'
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies (High System Stress)")

    anomalies = df[df['is_anomaly']]

    if not anomalies.empty:
        st.dataframe(
            anomalies[['diagnosis_group','year','hospital_pressure_index','z_score']]
            .sort_values('z_score', ascending=False)
        )
    else:
        st.write("No significant anomalies detected.")

with tab5:

    st.header("Derived Metrics Explanation")

    st.markdown("""
    This dashboard uses several derived variables to capture **system-level hospital pressure** during COVID.

    ---
    """)

    st.subheader("1. Emergency Rate")
    st.latex(r"Emergency\ Rate = \frac{Emergency\ Admissions}{Total\ Admissions}")
    st.write("""
    Represents dependency on urgent care pathways.  
    High values indicate reduced elective capacity.
    """)

    st.subheader("2. Elderly Ratio")
    st.latex(r"Elderly\ Ratio = \frac{Age\ 75+}{Total\ Admissions}")
    st.write("""
    Captures demographic pressure.  
    Higher values indicate more resource-intensive patients.
    """)

    st.subheader("3. Mean Length of Stay (LOS)")
    st.write("""
    Average duration of hospitalisation.  
    Longer stays → reduced bed turnover → system strain.
    """)

    st.subheader("4. Mean Waiting Time")
    st.write("""
    Indicates backlog and service delays.  
    Important during post-COVID recovery.
    """)

    st.subheader("5. Hospital Pressure Index (HPI)")
    st.latex(r"HPI = Emergency\ Rate \times Mean\ LOS \times Elderly\ Ratio")
    st.write("""
    Composite indicator capturing **multi-dimensional strain**:
    
    - Emergency dependence  
    - Patient complexity (age)  
    - Resource usage (length of stay)  

    High HPI → systemic stress concentration.
    """)

    st.markdown("---")

    st.subheader("Why These Metrics Matter")

    st.write("""
    These features allow detection of **non-obvious structural shifts**:
    
    - Not just *more patients*, but *different types of patients*
    - Transition from elective → emergency-heavy system
    - Increasing dominance of elderly, long-stay cases
    
    This is critical to understanding **COVID's true impact beyond raw admissions**.
    """)

# ---------------------------
# OPTIONAL: HIGHLIGHT SELECTED DIAGNOSIS
# ---------------------------
if selected_diag:
    st.subheader("Selected Diagnosis Overview")

    highlight_df = df[df.diagnosis_group.isin(selected_diag)]

    fig = px.line(
        highlight_df,
        x='year',
        y='hospital_pressure_index',
        color='diagnosis_group',
        markers=True
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)
