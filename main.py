
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv


load_dotenv()


# Mistral embedding model
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


parser = StrOutputParser()



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)



vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="medical_chatbot",
    embedding_function=embedding_model
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

prompt = ChatPromptTemplate.from_template(
    """
You are a medical information assistant.

Use ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Instructions:

- Give a clear, useful, and concise answer.
- Do not mention "Document 1", "Document 2", chunks, context, retrieval, or RAG.
- Organize the answer with useful headings when appropriate.
- Use bullet points for lists.
- Use bold text for important medical terms.
- Explain medical terminology in simple language.
- Include causes, symptoms, examination findings, diagnosis, and treatment
  only when that information is available in the context.
- Do not add information that is not present in the context.
- Do not diagnose the user.
- Do not prescribe medication or dosage.
- If the context does not contain enough information, clearly say:
  "I don't have enough information in the provided medical document."
- For emergency symptoms, advise the user to seek immediate professional
  medical care.
- End with:
  "This information is for educational purposes only and does not replace
  professional medical advice."

Answer naturally and directly.

Answer:
"""
)


def ask_medical_question(question: str):

   
    documents = retriever.invoke(question)


    
    


    
    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # Create  a prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })


    # Send prompt to Groq
    response = llm.invoke(messages)


    
    output = parser.invoke(response)


    return output


# # Chat loop
# while True:

#     user = input("\nYOU: ")

#     if user == "0":

#         print("Exiting...")

#         break


#     output = ask_medical_question(user)
#     print(output)

