from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import fitz



load_dotenv()
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)
pdf = fitz.open("./medicl.pdf")
documents = []

print("Successfully loaded..")

for page in pdf:
    text = page.get_text()


    if "acute rheumatic fever" in text.lower():

        print("Acute rheumatic fever is FOUND!")
        
    

    if text.strip():
        documents.append(text)

     

# print(documents[15][:1000])
        


print("Total pages:", len(documents))



text_spliter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap =200
)


chunks = text_spliter.create_documents(documents)
# print(chunks)


vector_Store = Chroma.from_documents(
    documents=chunks,
    collection_name="medical_chatbot",
    embedding=embedding_model,
    persist_directory="./chroma_db"
)




print("Ending")



# # Install

# # uv pip install -U langchain-pymupdf4llm
