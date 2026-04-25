"""
app.py  –  Streamlit dashboard for Self-Healing Network
--------------------------------------------------------
Run with:   streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Self-Healing Network", page_icon="🌐", layout="wide")

st.markdown("""
<style>
  .fail-box{background:#fff0f0;border-left:4px solid #e74c3c;padding:1rem;border-radius:6px;margin:.5rem 0;}
  .ok-box  {background:#f0fff4;border-left:4px solid #2ecc71;padding:1rem;border-radius:6px;margin:.5rem 0;}
</style>""", unsafe_allow_html=True)

st.title("🌐 Self-Healing Network – Predictive Failure Analysis")
st.markdown("*Select a model from the sidebar — every tab shows results for that model only.*")
st.divider()

# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest", "SVM", "Logistic Regression"],
    )
    n_samples = st.slider("Simulation samples", 3, 20, 6)
    run_btn   = st.button("🚀 Train & Analyse", use_container_width=True)

# ── cached pipeline (trains once, reuses forever) ─────────────────────────
@st.cache_resource(show_spinner="Training all 4 models … please wait")
def get_pipeline():
    for f in ["data/nsl_kdd_style.csv","data/cicids_style.csv","data/server_health.csv"]:
        if not os.path.exists(f):
            import generate_data  # noqa
            break

    from src.preprocessing import load_and_preprocess
    from src.train          import split_data, train_all
    from src.evaluate       import evaluate_models, plot_model_comparison

    X, y, features, scaler, df_raw = load_and_preprocess()
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained    = train_all(X_train, y_train)
    results_df, predictions = evaluate_models(trained, X_test, y_test)

    os.makedirs("plots", exist_ok=True)
    plot_model_comparison(results_df, save_path="plots/model_comparison.png")

    # ── per-model confusion matrix images ─────────────────────────────────
    for name, model in trained.items():
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal","Failure"],
                    yticklabels=["Normal","Failure"],
                    ax=ax, linewidths=0.5, cbar=False)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        fig.savefig(f"plots/cm_{name.replace(' ','_').lower()}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ── per-model feature importance images (tree models only) ────────────
    PALETTE = {"Random Forest":"#2ecc71", "XGBoost":"#e74c3c"}
    imp_data = {}
    for name in ("Random Forest", "XGBoost"):
        if name not in trained:
            continue
        model       = trained[name]
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:12]
        feat_names  = [features[i] for i in indices]
        feat_vals   = importances[indices]
        imp_data[name] = dict(zip(features, importances))

        fig, ax = plt.subplots(figsize=(7,6))
        ax.barh(feat_names[::-1], feat_vals[::-1],
                color=PALETTE[name], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{name} – Feature Importance", fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(f"plots/fi_{name.replace(' ','_').lower()}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    return trained, results_df, X_test, y_test, features, df_raw, scaler, imp_data, predictions


# ── main UI ───────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("trained"):
    st.session_state["trained"] = True

    with st.spinner("Running pipeline …"):
        (trained, results_df, X_test, y_test,
         features, df_raw, scaler, imp_data, predictions) = get_pipeline()

    sel      = model_choice
    safe_sel = sel.replace(" ", "_").lower()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Performance",
        "📈 Feature Importance",
        "🔮 Live Prediction",
        "🛠️  Self-Healing Demo",
    ])

    # ── Tab 1: metrics + confusion matrix for SELECTED model ──────────────
    with tab1:
        st.subheader(f"Performance – {sel}")
        row = results_df[results_df["Model"] == sel].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",  f"{row['Accuracy']:.4f}")
        c2.metric("Precision", f"{row['Precision']:.4f}")
        c3.metric("Recall",    f"{row['Recall']:.4f}")
        c4.metric("F1-Score",  f"{row['F1-Score']:.4f}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Confusion Matrix – {sel}**")
            cm_path = f"plots/cm_{safe_sel}.png"
            if os.path.exists(cm_path):
                st.image(cm_path, use_column_width=True)
        with col2:
            st.markdown("**All Models Comparison**")
            if os.path.exists("plots/model_comparison.png"):
                st.image("plots/model_comparison.png", use_column_width=True)

        st.divider()
        st.markdown("**Full results table**")
        # highlight selected model's row
        def highlight_selected(row):
            return ["background-color: #d5f5e3" if row["Model"]==sel else "" for _ in row]
        st.dataframe(results_df.style.apply(highlight_selected, axis=1), use_container_width=True)

    # ── Tab 2: feature importance for SELECTED model only ─────────────────
    with tab2:
        st.subheader(f"Feature Importance – {sel}")
        fi_path = f"plots/fi_{safe_sel}.png"
        if os.path.exists(fi_path):
            st.image(fi_path, use_column_width=True)
            if sel in imp_data:
                imp_df = pd.Series(imp_data[sel]).sort_values(ascending=False).reset_index()
                imp_df.columns = ["Feature","Importance"]
                st.dataframe(imp_df, use_container_width=True)
        else:
            st.warning(
                f"**{sel}** does not support feature importance.\n\n"
                "Feature importance is only available for **Random Forest** and **XGBoost**.\n\n"
                "Switch to one of those in the sidebar to see this chart."
            )

    # ── Tab 3: live prediction with SELECTED model ─────────────────────────
    with tab3:
        st.subheader(f"🔮 Live Prediction  ·  Model: **{sel}**")
        c1,c2,c3 = st.columns(3)
        with c1:
            latency         = st.slider("Latency (ms)",      0.0, 500.0, 20.0)
            packet_loss     = st.slider("Packet Loss",       0.0, 1.0,   0.01, 0.01)
            bandwidth       = st.slider("Bandwidth (Mbps)",  1.0, 200.0, 100.0)
            jitter          = st.slider("Jitter (ms)",       0.0, 100.0, 3.0)
        with c2:
            throughput      = st.slider("Throughput (Mbps)", 1.0, 200.0, 80.0)
            cpu_usage       = st.slider("CPU usage (%)",     0.0, 100.0, 30.0)
            memory_usage    = st.slider("Memory usage (%)",  0.0, 100.0, 40.0)
            disk_io         = st.slider("Disk I/O (%)",      0.0, 100.0, 20.0)
        with c3:
            network_errors   = st.slider("Network errors",   0, 200, 2)
            retransmissions  = st.slider("Retransmissions",  0, 100, 1)
            connection_drops = st.slider("Connection drops", 0,  50, 0)
            uptime_pct       = st.slider("Uptime (%)",       0.0, 100.0, 99.0)

        known = {"latency":latency,"packet_loss":packet_loss,"bandwidth":bandwidth,
                 "jitter":jitter,"throughput":throughput,"cpu_usage":cpu_usage,
                 "memory_usage":memory_usage,"disk_io":disk_io,
                 "network_errors":float(network_errors),
                 "retransmissions":float(retransmissions),
                 "connection_drops":float(connection_drops),"uptime_pct":uptime_pct}
        sample_dict = {f: known.get(f, 0.0) for f in features}

        if st.button("Predict now"):
            from src.self_healing import reason_and_heal
            model    = trained[sel]
            imp_dict = imp_data.get(sel, imp_data.get("XGBoost",
                       imp_data.get("Random Forest", {})))
            X_sc = scaler.transform(np.array([sample_dict[f] for f in features]).reshape(1,-1))[0]
            result = reason_and_heal(model, X_sc, sample_dict, features, imp_dict, verbose=False)

            if result["predicted_failure"]:
                st.markdown(
                    f'<div class="fail-box"><b>⚠️ FAILURE PREDICTED  ({sel})</b>'
                    f'<br>Confidence: {result["confidence"]*100:.1f}%</div>',
                    unsafe_allow_html=True)
                st.markdown("**Root-cause factors:**")
                for feat,val,label in result["reasons"]:
                    st.markdown(f"- **{label}** → `{val:.3f}`")
                st.markdown("**Healing actions:**")
                for a in result["healing_actions"]:
                    st.markdown(f"- {a}")
            else:
                st.markdown(
                    f'<div class="ok-box"><b>✅ NETWORK STABLE  ({sel})</b>'
                    f'<br>Confidence: {result["confidence"]*100:.1f}%</div>',
                    unsafe_allow_html=True)

    # ── Tab 4: self-healing demo with SELECTED model only ─────────────────
    with tab4:
        st.subheader(f"🛠️ Self-Healing Demo  ·  Model: **{sel}**")
        from src.self_healing import reason_and_heal
        model    = trained[sel]
        imp_dict = imp_data.get(sel, imp_data.get("XGBoost",
                   imp_data.get("Random Forest", {})))

        idx = np.random.default_rng(42).choice(len(X_test), n_samples, replace=False)
        for i, row_idx in enumerate(idx):
            sample_d = {col: float(df_raw.iloc[row_idx][col])
                        for col in features if col in df_raw.columns}
            result = reason_and_heal(model, X_test[row_idx], sample_d,
                                     features, imp_dict, verbose=False)
            conf_str = f" ({result['confidence']*100:.1f}%)" if result["confidence"] else ""
            lbl = "⚠️ FAILURE" if result["predicted_failure"] else "✅ STABLE"

            with st.expander(f"Sample #{i+1}  –  {lbl}{conf_str}"):
                if result["predicted_failure"]:
                    st.markdown("**Root-cause factors:**")
                    for feat,val,l in result["reasons"]:
                        st.markdown(f"- **{l}** → `{val:.3f}`")
                    st.markdown("**Healing actions:**")
                    for a in result["healing_actions"]:
                        st.markdown(f"- {a}")
                else:
                    st.success("Network is operating normally.")

else:
    st.info("👈 Click **Train & Analyse** in the sidebar to start.")
    st.markdown("""
    ### How it works
    1. Three datasets are merged (15,000 rows total).
    2. All four models are trained once and cached.
    3. **Switching the model in the sidebar instantly updates every tab** —
       confusion matrix, feature importance, live prediction, and the healing demo
       all reflect the selected model only.
    4. Feature importance is shown for tree-based models (Random Forest / XGBoost).
       For SVM and Logistic Regression, a clear message is shown instead.
    """)