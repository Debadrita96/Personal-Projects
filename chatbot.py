import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_DIR = "vectorstore"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY not set.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a domain support assistant.

Answer ONLY using the context below.

If the answer is not present in the documents, reply exactly with:

"I don’t have enough information in the provided documents."

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG Assistant ready. Type 'exit' to quit.\n")

    while True:

        question = input("User: ")

        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(question)

        print("Bot:", answer)
        print()

if __name__ == "__main__":
    main()
