import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

prompt=ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    llm = ChatGroq(model=llm)
    output_parser=StrOutputParser()
    chain=prompt |llm|output_parser
    answer=chain.invoke({"question":question})
    return answer

st.title("Enhaced Q&A chatbot with open source models")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("enter your open source model key",key="password")
llm=st.sidebar.selectbox("Select open source models",["gemma2-9b-it","Llama3-8b-8192"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


st.write("go ahead and write to ask questions")

user_input=st.text_input("you:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("pleae provide the quesry")