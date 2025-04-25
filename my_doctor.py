import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import gradio as gr

# Step 1: Load your internal medicine PDFs
pdf_dir = "data"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
all_docs = []

for pdf in pdf_files:
    path = os.path.join(pdf_dir, pdf)
    loader = PyPDFLoader(path)
    all_docs.extend(loader.load())

# Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

# Step 3: Create vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding)
db.save_local("my_doctor_db")

# Step 4: Load vector DB and connect to Ollama LLM
db = FAISS.load_local("my_doctor_db", embedding, allow_dangerous_deserialization=True)
llm = Ollama(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Step 5: Gradio UI
def ask_my_doctor(query):
    return qa_chain.run(query)

gr.Interface(
    fn=ask_my_doctor,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question..."),
    outputs="text",
    title="🩺 My Doctor - Internal Medicine Assistant",
    description="Ask questions based on internal medicine PDFs."
).launch(server_name="0.0.0.0", server_port=7860)
