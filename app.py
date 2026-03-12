import streamlit as st
from transformers import pipeline
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainable XLM-R Sentiment Analysis")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():

    model_dir = "bimobirra/explainable-xlmr-code-mixed-low-resource-lang"

    analyzer = pipeline(
        task="text-classification",
        model=model_dir,
        tokenizer=model_dir,
        return_all_scores=True
    )

    return analyzer

analyzer = load_model()

@st.cache_resource
def load_explainer(_analyzer):

    explainer = shap.Explainer(
        analyzer,
        shap.maskers.Text(r"\W+")
    )

    return explainer


explainer = load_explainer(analyzer)


st.title("Sentiment Analysis with Explainable XLM-R for Code-Mixed Low-Resource Languages")

st.markdown(
    "Input Text (Minangkabau Language, Bahasa Indonesia, English) to analyze"
)

text_input = st.text_area("Input Text", height=150)


if st.button("Analyze", type="primary"):

    if text_input.strip() == "":
        st.warning("Please input text!")

    else:

        with st.spinner("Analyzing sentiment..."):

            prediction = analyzer(text_input)[0]

            label = prediction["label"]
            score = prediction["score"] * 100

            if label == "LABEL_2":
                st.success(f"Positive sentiment | confidence: {score:.2f}%")

            elif label == "LABEL_1":
                st.info(f"Neutral sentiment | confidence: {score:.2f}%")

            else:
                st.error(f"Negative sentiment | confidence: {score:.2f}%")


        st.markdown("---")

        st.subheader("Word Contribution Analysis (Explainable AI)")

        st.write(
            "The visualization below shows how each word contributes to the model's sentiment prediction."
        )


        with st.spinner("Building explanation..."):

            shap_values = explainer([text_input])

            class_index = int(label.split("_")[1])

            words = shap_values.data[0]
            scores = shap_values.values[0][:, class_index]

            explanation = shap.Explanation(
                values=scores,
                base_values=shap_values.base_values[0][class_index],
                data=words,
                feature_names=words
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            shap.plots.bar(
                explanation,
                show=False,
                max_display=len(words)
            )

            st.pyplot(fig)

            plt.clf()