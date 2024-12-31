import os
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import validators,streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.set_page_config(page_title="langchain:summerize Text from given URL")
st.title("langchain:summerize the text")
st.subheader("summerize URL")
llm=ChatGroq(model="Llama3-8b-8192")
prompt_template="""
Provide a summary of following content with 300 words:
content:{text}
"""

prompt=PromptTemplate(
    input_variables=['text'],
    template=prompt_template
)
with st.sidebar:
    grow_api_key=st.text_input("groq api key",value='',type='password')

generic_url=st.text_input("URL",label_visibility='collapsed')

if st.button("summerize the content from YT"):
    if not grow_api_key.strip() or not generic_url.strip():
        st.error("please provide the info")
    elif not validators.url(generic_url):
        st.error("please enter valid URL")
    else:
        try:
            st.spinner("waiting..........")
            if "youtube" in generic_url:
                loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
            else:
                loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                             headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})

            docs=loader.load()
            chain=load_summarize_chain(llm=llm,chain_type='stuff',prompt=prompt)
            output_summary=chain.run(docs)
            st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception{e}")


