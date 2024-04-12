# commands:

# pip3 install -r requirements.txt
# python3 -m venv .venv
# source .venv/bin/activate
# streamlit run qa.py

import os
import dotenv
import streamlit as st
import openai
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# api key
dotenv.load_dotenv(".env", override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')

def clear_history():
    if 'history' in st.session_state:
        st.session_state['history'] = []

st.title('Q&A PDF files')
uploaded_file = st.file_uploader('', type=['pdf'])
add_file_btn = st.button('Upload', on_click=clear_history)
question = st.chat_input("What's your question?")

if uploaded_file and add_file_btn:

    with st.spinner('Working...'):
        # upload file
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./', uploaded_file.name)
        with open(file_name,'wb') as f:
            f.write(bytes_data)
        name, extension = os.path.splitext(file_name)
        docs = []
        if extension == '.pdf':
            # load file
            loader = PyPDFLoader(file_name)
            docs = loader.load()
        else:
            st.error('Document format is not supported')
        
        # chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        # embeddings
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=f'./{name}')
        # model
        llm = OpenAI(temperature=0)
        retriever = vector_store.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
        st.session_state['chain'] = chain # share the scope.
        st.success('File uploaded successfully')

if question and 'chain' in st.session_state:
    
    chain = st.session_state['chain']

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    response = chain.run({'question': question, 'chat_history': st.session_state['history']})
    st.session_state['history'].append((question, response))

    for qa in st.session_state['history']:
        user = st.chat_message("user")
        user.write(qa[0])
        bot = st.chat_message("assistant")
        bot.write(qa[1])
