
import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("models", "fake_news_model.pkl")

st.set_page_config(
    page_title="NewsVerify AI",
    page_icon="📰",
    layout="wide",
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #8b95a5;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📰 NewsVerify AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered news text classification</div>',
    unsafe_allow_html=True,
)

st.info(
    "This tool classifies text based on patterns learned from a labeled "
    "dataset. Its prediction is not proof that a news story is true or false."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run `python train_model.py` first."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.subheader("Analyze a news article")

news_text = st.text_area(
    "Paste a headline or news article",
    height=240,
    placeholder=(
        "Paste the headline and article text here..."
    ),
    help="Longer text may provide more useful classification signals.",
)

analyze = st.button(
    "Analyze News",
    type="primary",
    use_container_width=True,
)

if analyze:
    if len(news_text.strip()) < 40:
        st.warning(
            "Please enter at least 40 characters for analysis."
        )
    else:
        prediction = int(model.predict([news_text])[0])
        probabilities = model.predict_proba([news_text])[0]
        classes = list(model.classes_)

        fake_index = classes.index(1)
        real_index = classes.index(0)

        fake_probability = float(probabilities[fake_index])
        real_probability = float(probabilities[real_index])

        st.divider()
        st.subheader("Classification result")

        if prediction == 1:
            st.error("Model prediction: Fake-news-like")
        else:
            st.success("Model prediction: Real-news-like")

        st.caption(
            "These percentages indicate model scores, not verified "
            "probabilities that the article is true or false."
        )

        left, right = st.columns(2)

        with left:
            st.metric(
                "Real-news score",
                f"{real_probability:.1%}",
            )

        with right:
            st.metric(
                "Fake-news score",
                f"{fake_probability:.1%}",
            )

        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = ["Real-news score", "Fake-news score"]
        values = [real_probability, fake_probability]

        ax.barh(labels, values)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Model score")
        ax.set_title("Classification score comparison")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.warning(
            "Always verify important claims using reliable independent "
            "sources. This model can misclassify satire, opinion, "
            "breaking news, and unfamiliar writing styles."
        )

st.divider()
st.caption(
    "NewsVerify AI | TF-IDF + Logistic Regression | "
    "Educational machine-learning project"
)
