import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings 

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

llm=ChatGroq(model='Llama3-8b-8192')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=embeddings
        st.session_state.loader=PyPDFDirectoryLoader("research_papers") 
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents( st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


user_prompt=st.text_input("enter your query from your research papers")

if st.button("Document embedding"):
    create_vector_embedding()
    st.write("vector database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()

    response=retriever_chain.invoke({"input":user_prompt})

    print(f"Response Time: {time.process_time()-start}")

    st.write(response["answer"])

    with st.expander("Document similarity Search"):
        for i, docs in enumerate(response['context']):
            st.write(docs.page_content)
            st.write("------------------------------------")