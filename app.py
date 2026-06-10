from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            knowledge_base = FAISS.from_documents(chunks, embeddings)
            retriever = knowledge_base.as_retriever(search_kwargs={"k": 4})

            user_question = st.text_input("Ask a question about your PDF:")

            if user_question:
                with st.spinner("Thinking..."):
                    llm = ChatOpenAI(
                        model="openrouter/auto",
                        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                        openai_api_base="https://openrouter.ai/api/v1",
                        temperature=0
                    )

                    prompt = PromptTemplate.from_template("""
                    Use the following context to answer the question.
                    If you don't know the answer, just say you don't know.

                    Context: {context}
                    Question: {question}
                    Answer:
                    """)

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    response = chain.invoke(user_question)
                    st.write(response)

        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

if __name__ == '__main__':
    main()