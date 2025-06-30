import streamlit as st
from utils import process_pdf, get_answer

st.set_page_config(page_title="SGH Discharge Q&A Assistant")
st.title("SGH Discharge Note Q&A Assistant")

uploaded_file = st.file_uploader("Upload a discharge summary (PDF)", type=["pdf"])

question = st.text_input("Ask a question about the discharge note")  # ✅ 提前展示

if uploaded_file:
    with st.spinner("Processing document..."):
        retriever = process_pdf(uploaded_file)

    if question:
        with st.spinner("Generating answer..."):
            answer = get_answer(question, retriever)
            st.markdown("### 🧠 Answer")
            st.write(answer)
else:
    st.info("Please upload a PDF file to begin.")
