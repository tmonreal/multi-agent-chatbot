"""
Use this script to load the querying interface for my Flask app or CLI.
"""
import os
from dotenv import load_dotenv
from embeddings import SentenceTransformerEmbedding
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

load_dotenv()

def classify_query_target(llm, query: str):
    """
    Uses LLM to classify whether the query is about Trinidad, Jorge, both, or neither.
    Defaults to 'trinidad' if unclear.
    """
    classification_prompt = f"""
    You are a classifier that determines if the following question refers to one of the following people:
    - Trinidad Monreal
    - Jorge Valdez
    - Both
    - Neither

    Respond with exactly: 'trinidad', 'jorge', 'both', or 'none'.

    Examples:

    Q: Where did Jorge study?  
    A: jorge

    Q: What are her top skills?  
    A: trinidad

    Q: Where did they both study?  
    A: both

    Q: Who has more experience in data science?
    A: both

    Now classify this question:
    Q: {query}
    A:
    """

    raw_output = llm.invoke(classification_prompt)
    result = getattr(raw_output, "content", str(raw_output)).strip().lower()
    print("Raw result content:", result)

    for label in ["trinidad", "jorge", "both", "none"]:
        if result.startswith(label):
            return label

    print("⚠️ Classification unclear. Defaulting to: trinidad")
    return "trinidad"

def load_multi_agents():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Multilingual model because CVs are in spanish and english
    embedding_model = SentenceTransformerEmbedding(
        SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    retrievers = {
        "trinidad": PineconeVectorStore(
            index_name="tmonreal",
            embedding=embedding_model,
            namespace="espacio"
        ).as_retriever(),
        "jorge": PineconeVectorStore(
            index_name="jvaldez",
            embedding=embedding_model,
            namespace="espacio"
        ).as_retriever(),
    }

    groq_model = ChatGroq(
        api_key=GROQ_API_KEY,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        streaming=False
    )

    prompt = PromptTemplate(
        template="""
        Use the context below to answer the question. Be concise. If you don't know, say 'I don't know'.

        Context:
        {context}

        Question: {question}
        Helpful Answer:
        """,
        input_variables=["context", "question"]
    )

    qa_chains = {
        name: RetrievalQA.from_chain_type(
            llm=groq_model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        for name, retriever in retrievers.items()
    }

    return qa_chains, groq_model

def run_combined_rag(query: str, qa_chains: dict, llm, debug=False) -> str:
    # Retrieve relevant documents for each person
    docs_t = qa_chains["trinidad"].retriever.invoke(query)
    docs_j = qa_chains["jorge"].retriever.invoke(query)

    # Add source labels for llm to distinguish 
    for doc in docs_t:
        doc.page_content = f"Trinidad's CV:\n{doc.page_content}"
    for doc in docs_j:
        doc.page_content = f"Jorge's CV:\n{doc.page_content}"

    all_docs: list[Document] = docs_t + docs_j

    if debug:
        print("\n📄 Top retrieved chunks:")
        for i, doc in enumerate(all_docs):
            print(f"\n--- Chunk {i+1} ---\n{doc.page_content.strip()}")

    prompt = PromptTemplate(
        template="""
        Use the following context from two professional profiles to answer the question below.
        Be comparative when relevant, specific, and concise. If information is missing, say "I don't know".

        Context:
        {context}

        Question: {question}
        Helpful Answer:
        """,
        input_variables=["context", "question"],
    )

    stuff_chain = create_stuff_documents_chain(llm, prompt, document_variable_name="context")
    response = stuff_chain.invoke({"context": all_docs, "question": query})
    return response