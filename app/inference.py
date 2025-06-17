# Streamlit Ui or Inference Script
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./final_model_lora"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🚀 GenAI Fine-Tuned Model Inference")
st.write("Enter your instruction and input below:")

instruction = st.text_area("Instruction (e.g., Summarize this article, Translate this text):", height=100)
input_text = st.text_area("Input Text:", height=150)

if st.button("Generate Output"):
    if instruction.strip() == "" or input_text.strip() == "":
        st.warning("Please enter both instruction and input text.")
    else:
        full_input = f"{instruction.strip()}\n{input_text.strip()}"
        inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("📤 Model Output:")
        st.success(output_text)
