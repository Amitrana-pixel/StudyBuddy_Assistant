from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding model
embedding_mod = MistralAIEmbeddings(
    model="mistral-embed"
)

# Load vector database
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_mod
)

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4
    }
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

USE ONLY the provided context to answer the question.

If the answer is not present in the context, say:
'I could not find the answer in the document.'
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])

print("RAG is ready! Ask your question about the document.")
print("Type 'exit' to quit.")

while True:

    query = input("\nYour question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    # Combine retrieved text
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # Create prompt input
    prompt_input = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    result = llm.invoke(prompt_input)

    # Print answer
    print("\nAI Answer:\n")
    print(result.content)

    print("\n" + "=" * 50)