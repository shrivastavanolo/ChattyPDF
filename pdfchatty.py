import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title("ChattyPDF")
    st.markdown('''
                About:
                An LLM powered chatbot built using:
                - Streamlit
                - LangChain
                - OpenAI''')
    add_vertical_space(5)
    st.write("Made by [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)")

load_dotenv()
def main():
    st.header("Learn your PDF without reading all of it")

    pdf=st.file_uploader("Upload your PDF",type="pdf")
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                Vectorstore = pickle.load(f)
            # st.write("Embeddings Loaded from Disk")
        else:
            embeddings=OpenAIEmbeddings()
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(Vectorstore,f)
            # st.write("Embeddings Computation Completed")
    
        query=st.text_input("Ask question about your file: ")

        if query:
            docs=Vectorstore.similarity_search(query=query,k=3)
            llm=OpenAI(temperature=0,)
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            response=chain.run(input_documents=docs,question=query)
            st.write(response)

if __name__=='__main__':
    main()