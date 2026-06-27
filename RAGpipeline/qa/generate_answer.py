import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from retrieval.retrieve import retrieve_documents

# ---------------- LOAD ENV ---------------- #

load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGoogleGenerativeAI(

    model="gemini-2.5-flash",

    temperature=0.2
)

# ---------------- USER QUESTION ---------------- #

question = input("Enter your Question : ")

# ---------------- RETRIEVE ---------------- #

retrieved_chunks = retrieve_documents(question)

# ---------------- CONTEXT ---------------- #

context = "\n\n".join(

    chunk.page_content

    for chunk in retrieved_chunks

)

# ---------------- PROMPT ---------------- #

prompt = ChatPromptTemplate.from_template(
"""
You are an AI Question Answering Assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
reply exactly:

"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm

response = chain.invoke({

    "context": context,

    "question": question

})

print("\n")
print("="*70)
print("FINAL ANSWER")
print("="*70)

print(response.content)