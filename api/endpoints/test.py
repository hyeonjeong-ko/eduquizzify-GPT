from fastapi import FastAPI, UploadFile, File
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the GPT Summarization Server"}