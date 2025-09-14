from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class LotteryVisualizer:
    def plot_winning_numbers_trend(self, df: pd.DataFrame):
        vals = pd.DataFrame(df["winning_numbers"].tolist(), columns=[f"d{i+1}" for i in range(4)])
        vals["draw_date"] = df["draw_date"].values
        long = vals.melt(id_vars="draw_date", var_name="digit_pos", value_name="value")
        fig = px.line(long, x="draw_date", y="value", color="digit_pos", title="Digit values over time")
        return fig

    def plot_sum_distribution(self, df: pd.DataFrame):
        fig = px.histogram(df, x="sum", nbins=20, title="Sum distribution")
        return fig

    def plot_model_comparison(self, metrics: Dict[str, float]):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(metrics.keys()), y=list(metrics.values())))
        fig.update_layout(title="Model comparison (higher is better)", yaxis_title="Score")
        return fig

    def plot_feature_importance(self, importances: List[float], feature_names: List[str]):
        fig = go.Figure([go.Bar(x=feature_names, y=importances)])
        fig.update_layout(title="Feature importance")
        return fig
