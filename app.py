from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere

load_dotenv()

def main():
    st.set_page_config(page_title="Ask your PDF", page_icon="💬")
    st.header("Ask your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        with open("temp.pdf", "wb") as f:
            f.write(pdf.getbuffer())

        try:
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)

            # Runs locally, no API key needed
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            knowledge_base = FAISS.from_documents(chunks, embeddings)

            user_question = st.text_input("Ask a question about your PDF:")

            if user_question:
                with st.spinner("Thinking..."):
                    llm = ChatCohere(
                        model="command-r",
                        cohere_api_key=os.getenv("hYqzfjBj0CKScuqnp1Ne9NVAbrdcq6zlz5hkleWm"),
                        temperature=0
                    )

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=knowledge_base.as_retriever(search_kwargs={"k": 4}),
                        return_source_documents=False
                    )

                    response = qa_chain.invoke({"query": user_question})
                    st.write(response["result"])

        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

if __name__ == '__main__':
    main()