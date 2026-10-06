from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main import ask_medical_question


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():

    return {
        "message": "Welcome to our chatbot"
    }


@app.post("/chatbot")
async def chat_bot(question: str):

    answer = ask_medical_question(question)

    return answer