# commands:

# pip3 install -r requirements.txt
# streamlit run qa.py

import os
import dotenv
import streamlit as st
import openai
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# api key
dotenv.load_dotenv(".env", override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')

# load file
loader = PyPDFLoader('./bitcoin.pdf')
docs = loader.load()

# chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# embeddings
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./docs_db')

# ----------------------------- UI ----------------------------- 
st.title('Q&A PDF files')

# model
llm = OpenAI(temperature=0)
retriever = vector_store.as_retriever()
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
question = st.text_input("What's your question?")

if question:
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    response = chain.run({'question': question, 'chat_history': st.session_state['history']})
    st.session_state['history'].append((question, response))
    st.write(response)
