from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
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
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import boto3

from fastapi import APIRouter

router = APIRouter()


@router.get("/test-summarize")
async def test_summarize():
    return {"message": "Summarize endpoint is working!"}


"""
S3_url받으면 -> 파일 임베딩으로 변환하고 저장하는 함수
요약 요청->요약해주는 함수
퀴즈생성 요청->퀴즈카드생성함수
학습카드 요청->학습카드생성함수
"""
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# S3 클라이언트 설정 (boto3 사용)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)


# FastAPI 인스턴스 생성
# app = FastAPI()

# GPT 모델 설정
llm = ChatOpenAI(temperature=0.1, streaming=True)


class EmbeddingRequest(BaseModel):
    s3_url: str  # S3 URL
    section_id: str  # 섹션 ID


class SummarizeRequest(BaseModel):
    sourceType: str  # 요청 유형 (summary, quiz 등)
    sectionId: str  # 섹션 ID
    focusPrompt: Optional[str] = None  # 추가 요청 내용 (선택적)


def download_from_s3(s3_url: str, download_path: str):
    """S3에서 파일 다운로드"""
    try:
        # S3 URL에서 버킷 이름과 파일 경로 추출
        s3_parts = s3_url.replace("https://", "").split("/", 1)
        bucket_name = "eduquizzify-user-files"  # s3_parts[0]
        object_key = s3_parts[1]

        # S3에서 파일 다운로드
        s3_client.download_file(bucket_name, object_key, download_path)
        return download_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 파일 다운로드 오류: {str(e)}")


def create_embedding(directory_path, file_path):
    """임베딩을 생성하고 저장"""
    # 파일을 읽고 문서로 변환
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    # FAISS 벡터스토어에 임베딩 저장
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(directory_path)


@router.post("/embeddings")
async def create_embedding_api(request: EmbeddingRequest):
    """
    S3 URL을 받아 임베딩을 생성하는 API
    """

    # s3_client.download_file(
    #     "eduquizzify-user-files",  # S3 버킷 이름
    #     "test123_20240911171846_d2005c5c-0376-46d7-a0a6-dccc3b6ea993_bts.txt",  # S3에 있는 파일 경로
    #     "/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/api/v1/file_upload/test123_20240911171846_d2005c5c-0376-46d7-a0a6-dccc3b6ea993_bts.txt",  # 로컬에 저장할 전체 경로 (파일 이름 포함)
    # )

    try:
        # 섹션 ID 기반 디렉토리 경로 생성
        cache_dir = f"/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/{request.section_id}"

        # 이미 해당 섹션의 임베딩이 존재하는지 확인
        ## 캐시
        if os.path.exists(cache_dir):
            return {
                "message": "이미 임베딩이 생성된 섹션입니다.",
                "cache_dir": cache_dir,
            }

        # 캐시 디렉토리가 없으면 생성
        os.makedirs(cache_dir, exist_ok=True)

        # 파일을 다운로드할 로컬 경로 설정
        download_path = os.path.join(cache_dir, request.s3_url.split("/")[-1])

        print("download_path", download_path)

        # S3에서 파일 다운로드
        downloaded_file = download_from_s3(request.s3_url, download_path)

        print("파일 다운로드가 완료되었습니다.")

        # 임베딩 생성 및 저장
        create_embedding(cache_dir, downloaded_file)

        return {"message": "임베딩 생성 성공", "cache_dir": cache_dir}
    except Exception as e:
        # 오류 발생 시 캐시 디렉토리 삭제 (임시 파일 정리)
        if os.path.exists(cache_dir):
            # shutil.rmtree(cache_dir)
            raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")


def extract_docs_from_file(file_path):
    """S3에 업로드된 파일에서 텍스트를 추출하는 함수"""
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def summarize_by_map_reduce(docs, focus_prompt):
    """GPT를 사용하여 문서를 요약하는 함수 (Map-Reduce 방식 + Focus Prompt)"""

    # 요약을 요청하는 메시지에 focus_prompt를 포함
    if focus_prompt:
        map_prompt_template = f"""
        Summarize the following document chunk with a focus on: {focus_prompt}
        {{docs}}
        """
        reduce_prompt_template = f"""
        The following are summaries of document chunks with a focus on: {focus_prompt}
        {{docs}}
        Combine these summaries into a final concise summary.
        """
    else:
        # focus_prompt가 없을 때의 기본 요약 템플릿
        map_prompt_template = """
        Summarize the following document chunk:
        {docs}
        """
        reduce_prompt_template = """
        The following are summaries of document chunks:
        {docs}
        Combine these summaries into a final concise summary.
        """

    # Map 단계
    map_chain = LLMChain(
        llm=llm, prompt=PromptTemplate.from_template(map_prompt_template)
    )

    # Reduce 단계
    reduce_llm_chain = LLMChain(
        llm=llm, prompt=PromptTemplate.from_template(reduce_prompt_template)
    )
    reduce_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain, document_variable_name="docs"
    )

    # Map-Reduce 문서 체인 구성
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
    )

    # 문서 요약 생성
    summary = map_reduce_chain.run(docs)
    return summary


@router.post("/summary")
async def summarize_docs(request: SummarizeRequest):
    """
    S3 파일 URL을 받아 문서 요약을 생성하는 API
    """
    section_id = request.sectionId
    focus_prompt = request.focusPrompt
    embeddings_dir = f"/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/{section_id}"

    # 섹션 ID 폴더가 존재하는지 확인
    if not os.path.exists(embeddings_dir):
        raise HTTPException(
            status_code=404,
            detail=f"섹션 ID {section_id}에 해당하는 폴더가 존재하지 않습니다.",
        )

    try:
        # 이미 저장된 벡터 기반으로 작업 수행 (임베딩 불러오기)
        vectorstore = FAISS.load_local(embeddings_dir, OpenAIEmbeddings())

        # 벡터를 바탕으로 문서 불러오기 (docs로 변환)
        docs = vectorstore.similarity_search(
            "Bring all the contents of the document as much as possible."
        )

        print(docs)
        print("나 실행돼???")

        # 텍스트 요약 생성 (포커스 프롬프트 포함)
        summary = summarize_by_map_reduce(docs, focus_prompt)

        return summary  # {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
