import streamlit as st
from transformers import pipeline
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainable XLM-R Sentiment Analysis")

@st.cache_resource
def load_model():
    model_dir = "./model"

    analyzer = pipeline(
        task="text-classification",
        model=model_dir,
        tokenizer=model_dir
    )

    return analyzer

analyzer = load_model()

st.title("Sentiment Analysis with Explainable XLM-R for Code-Mixed Low-Resource Languages")
st.markdown("Input Text (Minangkabau Language, Bahasa Indonesia, English) to analyze")

text_input = st.text_area("Input Text", height=150)

if st.button("Analyze", type="primary"):
    if text_input.strip() == "":
        st.warning("Please input text!")

    else:
        with st.spinner("Analyzing..."):
            result = analyzer(text_input)[0]
            label = result["label"]
            score = result["score"] * 100

            if label == "LABEL_2":
                st.success(f"Positve, confidence score: {score:.2f}%")
                color = "success"
            elif label == "LABEL_1":
                st.info(f"Neutral, confidence score: {score:.2f}%")
            else:
                st.error(f"Negative, confidence score: {score:.2f}%")

        st.markdown("---")
        st.markdown("Word Analysis (Explainable AI)")
        st.write("Visualization below shows which word effect the AI's decision making")

        with st.spinner("Building Visualization"):
            explainer = shap.Explainer(analyzer)
            shap_values = explainer({text_input})

            class_index = int(label.split("_")[1])

            fig, ax = plt.subplots(figsize=(10,5))

            shap.plots.bar(shap_values[0, :, class_index], show=False)
            st.pyplot(fig)
            plt.clf()
