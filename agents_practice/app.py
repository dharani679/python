from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
app = FastAPI()

llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
    api_key=os.environ.get("GEMINI_API_KEY"),
    # system_prompt="You are a helpful assistant taht provides accurate and consise answers to all the questions asked by the user. You are a professional and an expert in all the fields of knowledge. You provide answers in a very simple and easy to understand manner."
)

class Question(BaseModel):
    question:str

@app.post("/chat")
async def chat(request: Question):
    user_question = request.question
    response = llm.invoke(user_question)
    
    return{
        "question": user_question,
        "response": response.content
    }
