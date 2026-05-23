import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

# Header
st.title("💬 AI Sentiment Analyzer")
st.markdown("Analyze customer reviews instantly using **DistilBERT** — a state-of-the-art NLP model.")
st.divider()

# Two tabs
tab1, tab2 = st.tabs(["📝 Single Review", "📂 Bulk CSV Analysis"])

# Tab 1 — Single Review
with tab1:
    st.subheader("Analyze a Single Review")
    user_input = st.text_area(
        "Enter a customer review:",
        placeholder="e.g. This product is amazing! I highly recommend it.",
        height=150
    )

    if st.button("Analyze", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                result = model(user_input)[0]
                label = result["label"]
                score = result["score"] * 100

            col1, col2 = st.columns(2)

            with col1:
                if label == "POSITIVE":
                    st.success(f"😊 {label}")
                else:
                    st.error(f"😞 {label}")

            with col2:
                st.metric("Confidence Score", f"{score:.1f}%")

            st.progress(result["score"])

# Tab 2 — Bulk CSV Analysis
with tab2:
    st.subheader("Analyze Multiple Reviews from CSV")
    st.markdown("Upload a CSV file with a column named **`review`**")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must have a column named 'review'")
        else:
            st.write(f"Found **{len(df)}** reviews. Analyzing...")

            with st.spinner("Analyzing all reviews..."):
                results = model(df["review"].tolist(), truncation=True)
                df["Sentiment"] = [r["label"] for r in results]
                df["Confidence"] = [f"{r['score']*100:.1f}%" for r in results]

            st.divider()

            # Metrics
            col1, col2, col3 = st.columns(3)
            positive = len(df[df["Sentiment"] == "POSITIVE"])
            negative = len(df[df["Sentiment"] == "NEGATIVE"])
            col1.metric("Total Reviews", len(df))
            col2.metric("😊 Positive", positive)
            col3.metric("😞 Negative", negative)

            st.divider()

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df["Sentiment"].value_counts().reset_index()
                fig = px.pie(
                    sentiment_counts,
                    values="count",
                    names="Sentiment",
                    color="Sentiment",
                    color_discrete_map={
                        "POSITIVE": "#2ecc71",
                        "NEGATIVE": "#e74c3c"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Sentiment Breakdown")
                fig = px.bar(
                    sentiment_counts,
                    x="Sentiment",
                    y="count",
                    color="Sentiment",
                    color_discrete_map={
                        "POSITIVE": "#2ecc71",
                        "NEGATIVE": "#e74c3c"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Results table
            st.subheader("📋 Detailed Results")
            st.dataframe(df, use_container_width=True)

            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

st.divider()
st.caption("Built with HuggingFace Transformers · DistilBERT · Streamlit · NIT Calicut")
