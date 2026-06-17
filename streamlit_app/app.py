"""
PoA in Congestion Games — Interactive Results Explorer
=========================================================

A thin interactive layer over the Week 1-6 experiment outputs. This app does
NOT run any new simulations or recompute any PoA values — it only loads and
displays results that already exist on disk (comparison plots, the
cross-agent summary CSV, and the written findings document).

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

from data_loader import (
    get_algorithms,
    get_metrics_row,
    get_networks,
    get_plot_path,
    load_findings_text,
    load_summary,
    split_findings_into_sections,
)

st.set_page_config(
    page_title="Price of Anarchy — Results Explorer",
    layout="wide",
)

st.title("Price of Anarchy in Congestion Games")
st.caption(
    "Interactive explorer for the bandit-agent study (ε-Greedy, UCB, Thompson Sampling) "
    "across Pigou, Braess, and Erdős–Rényi networks."
)

df = load_summary()
networks = get_networks(df)
algorithms = get_algorithms(df)

with st.sidebar:
    st.header("Select a result")
    selected_network = st.selectbox("Network", networks)
    selected_algorithm = st.selectbox("Algorithm", algorithms)
    st.markdown("---")
    st.caption(
        "Plots and numbers below are loaded directly from "
        "`results/comparison/`. Nothing here is recomputed."
    )

tab_overview, tab_findings = st.tabs(["Overview", "Findings"])

with tab_overview:
    col_plot, col_metrics = st.columns([2, 1])

    with col_plot:
        st.subheader(f"{selected_network} — all algorithms")
        plot_path = get_plot_path(selected_network)
        if plot_path is not None:
            st.image(str(plot_path), use_container_width=True)
        else:
            st.warning(f"No comparison plot found for {selected_network}.")

    with col_metrics:
        st.subheader(f"{selected_algorithm} on {selected_network}")
        row = get_metrics_row(df, selected_network, selected_algorithm)
        if row is not None:
            st.metric("Selfish PoA (baseline)", f"{row['selfish_poa']:.4f}")
            st.metric(
                "Final PoA (100% bandit fraction)",
                f"{row['final_poa']:.4f}",
                delta=f"{row['pct_change']:+.2f}%",
                delta_color="inverse",  # a decrease in PoA is the good direction
            )
            st.markdown(f"**Direction:** {row['direction']}")
            st.markdown(f"**Nash equilibrium type:** {row['nash_type']}")
        else:
            st.warning("No summary row found for this combination.")

    st.markdown("---")
    st.subheader("Full summary table")
    st.dataframe(df, use_container_width=True)

with tab_findings:
    st.subheader("Empirical findings")
    st.caption(
        "These findings are comparative by nature — most reference two or "
        "three networks at once — so rather than guess which one is "
        "'most relevant' to your current selection, all five are listed "
        "below for you to read directly."
    )
    full_findings = load_findings_text()
    sections = split_findings_into_sections(full_findings)
    for section in sections:
        st.markdown(section)
        st.markdown("---")
