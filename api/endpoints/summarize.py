from fastapi import FastAPI, UploadFile, File
from langchain.chains import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    StuffDocumentsChain,
    LLMChain,
)
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 블로그요약
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

# .env 파일에서 환경 변수 로드
load_dotenv()

# FastAPI 인스턴스 생성
app = FastAPI()

# GPT 모델 설정
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
)


# 요청 데이터 모델
class SummarizeRequest(BaseModel):
    sourceType: str
    data: str  # 링크 또는 텍스트 데이터


def extract_docs_from_file(file_path):
    """파일로부터 텍스트를 추출하는 함수 (PDF, TXT, DOCX 지원)"""
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def summarize_by_map_reduce(docs):
    """GPT를 사용하여 문서를 요약하는 함수 (Map-Reduce 방식)"""

    map_template = PromptTemplate.from_template(
        """
        Summarize the following document chunk: {docs}
        """
    )
    reduce_template = PromptTemplate.from_template(
        """
        The following are summaries of document chunks:
        {docs}
        Combine these summaries into a final concise summary.
        """
    )

    # Map 단계
    map_chain = LLMChain(llm=llm, prompt=map_template)

    # Reduce 단계
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_template)
    reduce_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain, document_variable_name="docs"
    )

    # Map-Reduce 문서 체인 구성
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
    )

    summary = map_reduce_chain.run(docs)
    return summary


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


def load_website(url):
    """URL에서 텍스트를 로드하고 문서로 분리하는 함수"""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=100,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


# 1. 각 소스 타입에 맞는 요약 처리 함수 정의
# @app.post("/summarize/")
async def summarize_docs():
    """문서를 업로드하여 요약을 생성하는 API"""
    # file_location = f"./files/{file.filename}"
    file_location = "/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/files/congi.txt"
    """
    try:
        # 파일을 열고 내용 읽기
        with open(file_location, "r", encoding="utf-8") as f:
            content = f.read()
        return {"status": "success", "content": content}  # 파일 내용을 응답으로 반환
    except FileNotFoundError:
        return {"status": "error", "message": "File not found."}
    """
    # 파일에서 docs 추출
    extracted_docs_chunks = extract_docs_from_file(file_location)

    # 텍스트 요약
    summary = summarize_by_map_reduce(extracted_docs_chunks)

    # 파일 삭제 (저장 공간 관리)
    # os.remove(file_location)

    return {"summary": summary}


# 블로그 자료 요약 API
@app.post("/summarize/blog")
async def summarize_blog(url: str):
    """블로그 자료 요약"""
    try:
        # 블로그 URL에서 문서 로드 및 벡터 스토어 생성
        retriever = load_website(url)

        # 관련 문서 검색
        # 관련 문서 검색
        relevant_docs = retriever.get_relevant_documents(
            """
            Please analyze the content of this blog post and extract the key points, 
            main arguments, and important takeaways. Focus on summarizing the overall 
            message, core themes, and any actionable insights provided in the post.
            """
        )

        # 관련 문서를 기반으로 요약
        summary = summarize_by_map_reduce(relevant_docs)  # Map-Reduce 방식 사용
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# async def summarize_pdf(file: UploadFile) -> str:
#     """PDF 파일 요약"""
#     # PDF 파일 읽기 처리 로직 추가
#     content = await file.read()
#     template = PromptTemplate(template="Summarize the following PDF content: {data}")
#     prompt = template.format(data=content.decode("utf-8"))
#     return llm(prompt)


# async def summarize_youtube(url: str) -> str:
#     """유튜브 영상 요약 (STT 등 활용 가능)"""
#     # 유튜브 링크에서 텍스트 추출 (STT, API 등 활용)
#     transcript = extract_transcript_from_youtube(url)
#     template = PromptTemplate(
#         template="Summarize the following YouTube transcript: {transcript}"
#     )
#     prompt = template.format(transcript=transcript)
#     return llm(prompt)


# 2. /summarize API
# @app.post("/summarize")
# async def summarize(data: SummarizeRequest, file: UploadFile = None):
#     if data.sourceType == "docs":
#         summary = await summarize_docs(data.data)
#     elif data.sourceType == "blog":
#         summary = await summarize_blog(data.data)
#     elif data.sourceType == "pdf" and file is not None:
#         summary = await summarize_pdf(file)
#     elif data.sourceType == "youtube":
#         summary = await summarize_youtube(data.data)
#     else:
#         return {"error": "Invalid source type or missing file"}

#     return {"summary": summary}
