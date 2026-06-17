"""
Results Loader for the PoA Streamlit Demo
===========================================

Loads existing experiment outputs (no new computation, no new simulations):
  - results/comparison/cross_agent_summary.csv  -> structured PoA metrics
  - results/comparison/comparison_*.png         -> per-network comparison plots
  - results/comparison/FINDINGS.md              -> written findings document

This module is pure plumbing: it reads what Week 6 already produced and
exposes it in a form the Streamlit app can render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARISON_DIR = REPO_ROOT / "results" / "comparison"
SUMMARY_CSV = COMPARISON_DIR / "cross_agent_summary.csv"
FINDINGS_MD = COMPARISON_DIR / "FINDINGS.md"

# Maps the network names used in the summary CSV to their plot files.
NETWORK_TO_PLOT = {
    "Pigou": COMPARISON_DIR / "comparison_pigou.png",
    "Braess": COMPARISON_DIR / "comparison_braess.png",
    "ER Graph": COMPARISON_DIR / "comparison_er_graph.png",
}


def load_summary() -> pd.DataFrame:
    """Load the cross-agent summary table (algorithm x network x PoA metrics)."""
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at {SUMMARY_CSV}. "
            "Run generate_summary_table.py first."
        )
    return pd.read_csv(SUMMARY_CSV)


def get_networks(df: pd.DataFrame) -> list[str]:
    """Unique network names available in the summary table, in a fixed order."""
    preferred_order = ["Pigou", "Braess", "ER Graph"]
    available = set(df["network"].unique())
    return [n for n in preferred_order if n in available]


def get_algorithms(df: pd.DataFrame) -> list[str]:
    """Unique algorithm names available in the summary table, in a fixed order."""
    preferred_order = ["ε-Greedy", "UCB", "Thompson Sampling"]
    available = set(df["algorithm"].unique())
    return [a for a in preferred_order if a in available]


def get_metrics_row(df: pd.DataFrame, network: str, algorithm: str) -> Optional[pd.Series]:
    """Fetch the single summary row for a given network + algorithm pair."""
    match = df[(df["network"] == network) & (df["algorithm"] == algorithm)]
    if match.empty:
        return None
    return match.iloc[0]


def get_plot_path(network: str) -> Optional[Path]:
    """Path to the existing comparison plot for a given network, if it exists."""
    path = NETWORK_TO_PLOT.get(network)
    if path is not None and path.exists():
        return path
    return None


def load_findings_text() -> str:
    """Raw markdown text of the findings document."""
    if not FINDINGS_MD.exists():
        return "_FINDINGS.md not found._"
    return FINDINGS_MD.read_text(encoding="utf-8")


def split_findings_into_sections(full_text: str) -> list[str]:
    """
    Split the findings document into individual '## Finding N — ...' sections
    plus the leading title block. Used so the app can let a person browse
    findings one at a time instead of guessing which one is "most relevant"
    to a given network — the findings are comparative by nature (most
    mention two or three networks at once), so an automatic best-match
    heuristic would be guessing, not informing.
    """
    parts = full_text.split("## Finding")
    header = parts[0]
    findings = ["## Finding" + p.split("\n---")[0] for p in parts[1:]]
    return [header] + findings if header.strip() else findings
