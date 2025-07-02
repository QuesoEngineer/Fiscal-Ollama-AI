from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
import time
if __name__ == '__main__':
    # Load and split documents
    loader = TextLoader("./docs/finance_knowledge.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create local embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize Ollama LLM
    llm = OllamaLLM(model="phi3")

    # Custom prompt template to focus on financial advice
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are a helpful financial advisor chatbot.
    Use the provided context to answer questions about budgeting, investing, and personal finance.
    If you are unsure, say "I am not certain—please consult a certified financial professional."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )

    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Chat loop
    print("Personal Finance Advice Bot (Local). Type 'exit' to quit.")
    console = Console()
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        with console.status("[bold cyan]Generating response..."):
            result = qa_chain.invoke({"query": query})

        console.print("[green]Done![/green]")
        console.print(result['result'])


