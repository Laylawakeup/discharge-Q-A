import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch


# ---------- 1. Read PDF ----------
def load_pdf(file):
    """Read an uploaded PDF file-like object and return LangChain Documents."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents


# ---------- 2. Vector ----------
def create_vectorstore(documents):
    """Split documents, embed them and build a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


# ---------- 3.  LLM ----------
def create_llm():
    """Return a lightweight HuggingFace LLM wrapped for LangChain."""
    model_name = "microsoft/DialoGPT-medium"  

    text_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm


# ---------- 4. Streamlit Call ----------
def process_pdf(file, *, k: int = 4):
    """
    One-stop helper called by `app.py`.
    Steps:
      1. read PDF
      2. build vector store
      3. convert to retriever
    Returns:
      retriever – ready for Retrieval-QA.
    """
    documents = load_pdf(file)
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


# ---------- 5. Q&A ----------
def get_answer(question, retriever):
    """Run Retrieval-QA chain and return the answer text."""
    try:
        llm = create_llm()
        qa_chain = RetrievalQA.from_chain_type(
