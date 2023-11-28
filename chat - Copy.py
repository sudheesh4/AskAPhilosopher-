# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 03:21:28 2023

@author: sud4d
"""
import torch
import streamlit as st 
import os
import time
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import copy
import langchain
langchain.debug=True
#print(torch.cuda.is_available())
#torch.cuda.set_device(torch.device("cuda:0"))
#torch.cuda.empty_cache()

#import google.generativeai as palm
PALM_API="Add-PaLM-API-key"
#palm.configure(api_key=PALM_API)
from langchain.llms import GooglePalm


#llm=GooglePalm(google_api_key=PALM_API,temperature=0.003)


device='cpu'
NAMES=["Weil","Wittgenstein","Heidegger","Aristotle"]
FILE_DIRS=["/weil","/wittg","/heid","/arist"]



st.set_page_config(layout='wide')


choice=1
option = st.selectbox('Whom would you like to contact?',NAMES)

for i in range(len(NAMES)):
    if NAMES[i].find(option) != -1:
        choice=i
        break
#choice=(NAMES==option)
#print(choice)
#choice=1
st.title('Ask '+NAMES[choice])

@st.cache_resource
def getmodel(choice):
    from langchain.llms import CTransformers
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.chains import RetrievalQAWithSourcesChain
    
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader('./summaries'+FILE_DIRS[choice], glob="*.txt",loader_cls=TextLoader,loader_kwargs=text_loader_kwargs)#,encoding='utf-8-sig')
    
    
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],
                                              chunk_size=500,
                                              chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device})
    
   
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss")
    

    
    
    template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    # load the language model
    #llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q2_K.bin',
    #                    model_type='llama',
     #                   config={'max_new_tokens': 512, 'temperature': 0.001,'context_length' : 4000})
    llm=GooglePalm(google_api_key=PALM_API,temperature=0.003,maxOutputTokens=2000)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device})
    db = FAISS.load_local("faiss", embeddings)

    retriever = db.as_retriever(search_kwargs={'k': 5})
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                          chain_type='stuff',
                                          retriever=retriever,
                                          return_source_documents=False,
                                          chain_type_kwargs={'prompt': prompt},
                                        verbose=True)
    #qa_llm=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=retriever,verbose=True)
    
    # prompt = "Who is Martin Heidegger?"
    # output = qa_llm({'query': prompt})
    # print(output["result"])
    print("%%%%%%%%%%%%%%%$$$$$$$$$$$$$$$$")
    return qa_llm


qa_llm=getmodel(choice)
print(">>>>>>>>>######>>>>>>>>")

#####################################
#############################################
######################################




prompt = st.text_input('Enter question')


if prompt: 
    #response =ps.searchscraper(prompt)#agent_executor.run(prompt)
    #print(searchscaper("Hume"))
    #answer=response[0]
    #print(len(answer))
    #answer=llmpipeline(answer[0:5000])
    print("RESPONDING")
    t1=time.time()
    answer=qa_llm({'query': prompt})#["result"]
    #answer=qa_llm({"question":prompt},return_only_outputs=True)
    print(answer)
    print(time.time()-t1)
    st.write(answer)
    print("RESPONDED")
