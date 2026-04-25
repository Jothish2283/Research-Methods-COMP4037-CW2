import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("nhs_clean_final.csv")

data = load_data()

st.title("NHS COVID Impact: Structural Shift in Hospital Demand")

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(data.year.min()),
    int(data.year.max()),
    (2019, 2024)
)

selected_phase = st.sidebar.multiselect(
    "COVID Phase",
    options=data.covid_phase.unique(),
    default=data.covid_phase.unique()
)

# Apply filters
df = data[
    (data.year >= year_range[0]) &
    (data.year <= year_range[1]) &
    (data.covid_phase.isin(selected_phase))
]

# ---------------------------
# 🌟 HERO VISUAL: PARALLEL COORDINATES
# ---------------------------
st.header("System-Level Structural Shift (Hero Visualization)")

fig_parallel = px.parallel_coordinates(
    df,
    dimensions=[
        'emergency_rate',
        'elderly_ratio',
        'mean_los',
        'mean_wait',
        'hospital_pressure_index'
    ],
    color='year',
    color_continuous_scale=px.colors.diverging.Tealrose
)

st.plotly_chart(fig_parallel, use_container_width=True)

st.markdown("""
**Interpretation:**  
Observe how COVID years cluster toward high emergency dependence, higher elderly ratios, and longer hospital stays — indicating systemic pressure.
""")

# ---------------------------
# 🔁 SUPPORTING 1: SANKEY
# ---------------------------
st.header("Shift from Planned to Emergency Admissions")

df_sankey = df.groupby('covid_phase').agg({
    'emergency': 'sum',
    'admissions': 'sum'
}).reset_index()

labels = ['Pre-COVID', 'COVID Peak', 'Post-COVID', 'Emergency', 'Non-Emergency']

source = [0, 1, 2, 0, 1, 2]
target = [3, 3, 3, 4, 4, 4]

values = [
    df_sankey.loc[df_sankey.covid_phase == 'pre_covid', 'emergency'].sum(),
    df_sankey.loc[df_sankey.covid_phase == 'covid_peak', 'emergency'].sum(),
    df_sankey.loc[df_sankey.covid_phase == 'post_covid', 'emergency'].sum(),

    df_sankey.loc[df_sankey.covid_phase == 'pre_covid', 'admissions'].sum() -
    df_sankey.loc[df_sankey.covid_phase == 'pre_covid', 'emergency'].sum(),

    df_sankey.loc[df_sankey.covid_phase == 'covid_peak', 'admissions'].sum() -
    df_sankey.loc[df_sankey.covid_phase == 'covid_peak', 'emergency'].sum(),

    df_sankey.loc[df_sankey.covid_phase == 'post_covid', 'admissions'].sum() -
    df_sankey.loc[df_sankey.covid_phase == 'post_covid', 'emergency'].sum(),
]

fig_sankey = go.Figure(go.Sankey(
    node=dict(label=labels),
    link=dict(source=source, target=target, value=values)
))

st.plotly_chart(fig_sankey, use_container_width=True)

st.markdown("""
**Interpretation:**  
COVID caused a structural reallocation — suppressing planned care while increasing emergency dependency.
""")

# ---------------------------
# 🌳 SUPPORTING 2: TREEMAP
# ---------------------------
st.header("Diagnosis Burden by COVID Phase")

fig_tree = px.treemap(
    df,
    path=['covid_phase', 'icd_code'],
    values='admissions',
    color='elderly_ratio',
    color_continuous_scale='RdBu'
)

st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("""
**Interpretation:**  
Certain diagnoses become more elderly-dominated during COVID, reflecting selective hospitalisation pressure.
""")
