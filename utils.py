from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings 



import tempfile

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 💡 使用小批量嵌入以规避速率限制
    def safe_embed(texts, embedder, batch_size=1, delay=1.0):
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors.extend(embedder.embed_documents(batch))
            time.sleep(delay)
        return vectors

    embedded_docs = safe_embed(docs, embeddings, batch_size=1, delay=1.0)
    vectorstore = FAISS.from_documents(docs, embedded_docs)

    retriever = vectorstore.as_retriever()
    return retriever

def get_answer(question, retriever):
    llm = ChatOpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(question)
    return result

import time

