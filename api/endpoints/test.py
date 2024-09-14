from fastapi import FastAPI, UploadFile, File
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the GPT Summarization Server"}
