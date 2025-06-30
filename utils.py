import tempfile
import time
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  
from langchain.embeddings import HuggingFaceEmbeddings  

# ✅ Safe batching for any embedding model
def safe_embed(docs, embedder, batch_size=1, delay=1.0):
    vectors = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        text_batch = [doc.page_content for doc in batch]
        vectors.extend(embedder.embed_documents(text_batch))
        time.sleep(delay)
    return vectors

# ✅ Main pipeline: PDF → chunks → embeddings → retriever
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embedded_docs = safe_embed(docs, embeddings, batch_size=1, delay=0.5)

    vectorstore = FAISS.from_documents(docs, embedded_docs)
    retriever = vectorstore.as_retriever()
    return retriever

# ✅ Use retriever to answer question with LLM
def get_answer(question, retriever):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(question)
    return result
