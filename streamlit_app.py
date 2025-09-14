import io
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

from data_handler import LotteryDataHandler, LotteryAPI
from feature_engineering import FeatureEngineer
from model_trainer import LotteryModelTrainer, best_overlap
from visualization import LotteryVisualizer
from utils import to_digits, digits_to_str, multiset_overlap, greedy_multiset_mapping, ensure_state_dir, load_json, save_json

# Page configuration
st.set_page_config(page_title="Lottery Prediction Dashboard", page_icon="🎲", layout="wide")
st.title("🎲 Lottery Prediction Dashboard (Pick 4)")

# State/boosts
STATE_DIR = ".p4_dashboard_state"
BOOST_PATH = f"{STATE_DIR}/boost_counts.json"
ensure_state_dir(STATE_DIR)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Historical Analysis", "Predictions", "Model Performance"], index=0)

st.sidebar.header("Data source")
use_api = st.sidebar.toggle("Use API (demo)", value=False)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

st.sidebar.header("Prediction settings")
mutation_strength = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.4, 0.05)
num_predictions = st.sidebar.select_slider("Number of predictions", options=[10, 20, 30, 50, 100], value=30)

st.sidebar.header("Learning/Feedback")
enable_boosts = st.sidebar.checkbox("Enable feedback boosts", value=False)
boost_weight = st.sidebar.slider("Boost weight", 0.0, 5.0, 1.0, 0.1)
success_threshold = st.sidebar.selectbox("Success threshold (positionless overlap)", options=[3, 4], index=0)

# Initialize components
api = LotteryAPI()
data_handler = LotteryDataHandler(api=api)
feature_engineer = FeatureEngineer()
visualizer = LotteryVisualizer()
model_trainer = LotteryModelTrainer()

@st.cache_data(show_spinner=False)
def load_data_cached(use_api_flag: bool, uploaded_csv_bytes: Optional[bytes]) -> pd.DataFrame:
    bio = io.BytesIO(uploaded_csv_bytes) if uploaded_csv_bytes else None
    return data_handler.get_processed_data(use_api_flag, bio)

# Load data
try:
    uploaded_bytes = uploaded.read() if uploaded else None
    data = load_data_cached(use_api, uploaded_bytes)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Prepare numeric series
numbers_list: List[Tuple[int, ...]] = [to_digits(t, 4) for t in data["winning_numbers"].tolist()]
if len(numbers_list) < 5:
    st.warning("Limited history; results may be noisy.")

# Fit model on history up to last draw
hist_for_fit = numbers_list[:-1] if len(numbers_list) > 1 else numbers_list
model_trainer.fit(hist_for_fit)

# Pages
if page == "Historical Analysis":
    st.header("Historical Lottery Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(visualizer.plot_winning_numbers_trend(data), use_container_width=True)
    with c2:
        st.plotly_chart(visualizer.plot_sum_distribution(data), use_container_width=True)

    st.subheader("Recent Drawing History")
    st.dataframe(data[["draw_date", "winning_numbers", "sum"]].tail(15).sort_values("draw_date", ascending=False), use_container_width=True)

elif page == "Predictions":
    st.header("Lottery Number Predictions")

    latest_numbers = numbers_list[-1] if numbers_list else (1, 2, 3, 4)
    st.subheader("Most Recent Winning Numbers")
    st.write(f"Draw Date: {data.iloc[-1]['draw_date'].strftime('%Y-%m-%d')}")
    st.write("Numbers:", latest_numbers)
    st.write("Sum:", sum(latest_numbers))

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            boosts: Dict[str, float] = load_json(BOOST_PATH) if enable_boosts else {}
            preds = model_trainer.generate_predictions(
                last_draw=latest_numbers,
                n=num_predictions,
                mutation_strength=mutation_strength,
                boosts=boosts,
                boost_weight=boost_weight,
            )
            preds_str = [digits_to_str(p) for p in preds]
            st.success("Predictions generated.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Top predictions")
                st.write(preds_str)
                st.download_button("Download predictions (CSV)",
                                   pd.DataFrame({"prediction": preds_str}).to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv",
                                   mime="text/csv")

            with col2:
                st.markdown("Teach the model (manual success marking)")
                seed_input = st.text_input("Seed (previous draw)", value=digits_to_str(numbers_list[-2]) if len(numbers_list) >= 2 else "")
                actual_input = st.text_input("Actual next draw", value="")
                use_current_preds = st.checkbox("Use current predictions", value=True)
                pasted_preds = st.text_area("Or paste predictions (comma-separated)", value="")

                if st.button("Mark success (positionless)"):
                    if not actual_input.strip():
                        st.warning("Enter the actual next draw.")
                    else:
                        seed_t = to_digits(seed_input, 4)
                        actual_t = to_digits(actual_input, 4)
                        pred_list = preds_str if use_current_preds else [p.strip() for p in pasted_preds.split(",") if p.strip()]
                        if not pred_list:
                            st.warning("No predictions provided.")
                        else:
                            overlaps = [multiset_overlap(to_digits(p, 4), actual_t) for p in pred_list]
                            best_ov = max(overlaps)
                            st.write(f"Best overlap vs actual: {best_ov}")
                            if enable_boosts and best_ov >= success_threshold:
                                best_idx = int(np.argmax(overlaps))
                                best_pred_t = to_digits(pred_list[best_idx], 4)
                                pairs = greedy_multiset_mapping(seed_t, best_pred_t)
                                b = load_json(BOOST_PATH)
                                for x, y in pairs:
                                    key = f"{x}->{y}"
                                    b[key] = b.get(key, 0.0) + 1.0
                                save_json(BOOST_PATH, b)
                                st.success("Success recorded and boosts updated.")
                            else:
                                st.info("Recorded result. Boosts not applied (disabled or below threshold).")

else:
    st.header("Model Performance")
    # Rolling one-step-ahead evaluation with success ≥3 and =4
    rows = []
    boosts_eval: Dict[str, float] = load_json(BOOST_PATH) if enable_boosts else {}
    for t in range(1, len(numbers_list)):
        hist = numbers_list[:t]
        model_trainer.fit(hist[:-1] if len(hist) > 1 else hist)
        last = hist[-1]
        preds = model_trainer.generate_predictions(last, n=30, mutation_strength=0.3, boosts=boosts_eval, boost_weight=boost_weight)
        actual = numbers_list[t]
        ov = best_overlap(preds, actual)
        rows.append({"t": t, "best_overlap": ov, "success3": int(ov >= 3), "success4": int(ov == 4)})
    perf = pd.DataFrame(rows)
    if perf.empty:
        st.info("Not enough data for evaluation.")
    else:
        perf["rolling_success3"] = perf["success3"].rolling(50, min_periods=1).mean()
        perf["rolling_success4"] = perf["success4"].rolling(50, min_periods=1).mean()
        st.dataframe(perf.tail(20), use_container_width=True)
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success3"], name="Rolling success ≥3"))
            fig.add_trace(go.Scatter(x=perf["t"], y=perf["rolling_success4"], name="Rolling success =4"))
            fig.update_layout(title="Rolling success rates", xaxis_title="t (step)", yaxis_title="Rate")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.subheader("Boosts status")
    current_boosts = load_json(BOOST_PATH)
    total_pseudo = sum(current_boosts.values()) if current_boosts else 0
    st.write(f"Stored boosts: {total_pseudo} total pseudo-counts across {len(current_boosts)} transitions")
    if st.button("Clear boosts"):
        save_json(BOOST_PATH, {})
        st.success("Boosts cleared.")
