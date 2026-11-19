import ast
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats


st.set_page_config(page_title="MovieIQ", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))


# ---------- Data and model ----------
def first_genre(s):
    genres = ast.literal_eval(s)
    return genres[0]["name"] if genres else "Unknown"


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "movies.csv"))
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genre"] = df["genres"].apply(first_genre)
    return df


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE, "model.pkl"))
    columns = joblib.load(os.path.join(BASE, "model_columns.pkl"))
    return model, columns


df = load_data()
model, model_columns = load_model()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_genres = sorted(df["genre"].unique())
selected = st.sidebar.multiselect("Genre", all_genres, default=all_genres)
min_vote = st.sidebar.slider("Minimum vote average", 3.0, 9.0, 3.0, 0.1)

filtered = df[df["genre"].isin(selected) & (df["vote_average"] >= min_vote)]

# ---------- Header ----------
st.title("MovieIQ: Predictive Analytics on Film Success")
st.caption("A movie is successful when revenue > budget.")

if filtered.empty:
    st.warning("No movies match these filters. Try widening them.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Movies", f"{len(filtered):,}")
c2.metric("Success rate", f"{filtered['success'].mean():.1%}")
c3.metric("Average budget", f"${filtered['budget'].mean() / 1e6:,.0f}M")

tab_eda, tab_stats, tab_predict = st.tabs(["Explore", "Statistical tests", "Predict"])

# ---------- Tab 1: EDA ----------
with tab_eda:
    left, right = st.columns(2)

    with left:
        st.subheader("Budget vs. revenue")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=filtered, x="budget", y="revenue", hue="success", alpha=0.5, ax=ax)
        top = filtered["budget"].max()
        ax.plot([0, top], [0, top], "k--", label="Break-even")
        ax.legend()
        st.pyplot(fig)

    with right:
        st.subheader("Success rate by genre")
        rate = filtered.groupby("genre")["success"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=rate.values, y=rate.index, ax=ax)
        ax.set_xlabel("Success rate")
        st.pyplot(fig)

    st.subheader("Popularity, runtime and vote average by success")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(data=filtered, x="success", y=col, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation heatmap")
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(filtered[num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

# ---------- Tab 2: statistical tests ----------
with tab_stats:
    st.subheader("T-tests (Welch): successful vs. unsuccessful movies")
    succ = filtered[filtered["success"] == 1]
    fail = filtered[filtered["success"] == 0]

    if len(succ) < 2 or len(fail) < 2:
        st.info("Need at least 2 successful and 2 unsuccessful movies to run the T-tests.")
    else:
        rows = []
        for col in ["popularity", "runtime", "vote_average"]:
            t, p = stats.ttest_ind(succ[col], fail[col], equal_var=False)
            rows.append({"Feature": col, "t": round(t, 3), "p-value": round(p, 4),
                         "Significant (p < 0.05)": p < 0.05})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("H0: the mean is the same for successful and unsuccessful movies.")

    st.subheader("Chi-square test: genre vs. success")
    ct = pd.crosstab(filtered["genre"], filtered["success"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        st.info("Need at least 2 genres and both outcomes to run the Chi-Square test.")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        st.write(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
        if expected.min() < 5:
            st.warning("Some expected counts are below 5, so this result is unreliable.")
        st.caption("H0: genre and success are independent.")

# ---------- Tab 3: prediction ----------
with tab_predict:
    st.subheader("Predict a movie's success")
    st.warning(
        "This model performed no better than chance on held-out data "
        "(test accuracy 0.74 vs. 0.81 baseline, AUC 0.52). "
        "Treat its output as a demo, not as advice."
    )

    p1, p2 = st.columns(2)
    with p1:
        budget_m = st.slider("Budget (millions $)", 1, 200, 100)
        popularity = st.slider("Popularity", 1.0, 100.0, 50.0)
        runtime = st.slider("Runtime (minutes)", 80, 179, 120)
    with p2:
        vote = st.slider("Average vote", 3.0, 9.0, 6.0, 0.1)
        genre = st.selectbox("Genre", all_genres)

    if st.button("Predict"):
        row = pd.DataFrame(0, index=[0], columns=model_columns)
        row["budget"] = budget_m * 1_000_000
        row["popularity"] = popularity
        row["runtime"] = runtime
        row["vote_average"] = vote
        row[f"genre_{genre}"] = 1

        pred = model.predict(row)[0]
        prob = model.predict_proba(row)[0][1]

        if pred == 1:
            st.success(f"Prediction: SUCCESS (model probability {prob:.0%})")
        else:
            st.error(f"Prediction: NOT a success (model probability of success {prob:.0%})")