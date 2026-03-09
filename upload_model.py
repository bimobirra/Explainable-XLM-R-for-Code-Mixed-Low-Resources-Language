from huggingface_hub import login, HfApi

login("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")
api = HfApi()
print("Memulai proses upload ke Hugging Face...")
api.upload_folder(
    folder_path="./model",
    repo_id="bimobirra/explainable-xlmr-code-mixed-low-resource-lang",
    repo_type="model"
)
print("Upload Selesai! Model siap digunakan di Streamlit.")