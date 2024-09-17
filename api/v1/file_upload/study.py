import os
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import glob
import logging

from fastapi import APIRouter

router = APIRouter()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 환경 변수 로드
load_dotenv()


# 학습 카드 요청 데이터 모델 정의
class StudyRequest(BaseModel):
    section_id: str  # 섹션 ID (필수)
    focus_prompt: Optional[str] = None  # 사용자가 추가한 프롬프트 (선택 사항)


# 학습 카드 응답 데이터 모델 정의
class StudyResponse(BaseModel):
    learning_cards: List[dict]


# 학습 카드 생성 Prompt
learning_cards_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that acts as a teacher.

            Based ONLY on the following context, create 5 (FIVE) learning cards to help the user study the material.

            Each learning card should contain a key fact or concept derived from the text.

            Format each card as a simple statement followed by a question in parentheses.

            Separate each learning card with a newline character.

            For example, if the context is about Ahn Jung-geun, the output should look like:

            Ahn Jung-geun assassinated Japanese Resident General Ito Hirobumi at Harbin Station in 1909. (Who did Ahn Jung-geun assassinate at Harbin Station in 1909?)
            He organized the Korean Independence Army and fought against Japanese imperialism. (What did Ahn Jung-geun organize to fight against Japanese imperialism?)
            Ahn Jung-geun was sentenced to death in a Japanese court in 1910 and was executed. (What was Ahn Jung-geun's fate after his trial in 1910?)
            His actions greatly influenced the Korean independence movement. (How did Ahn Jung-geun's actions impact the Korean independence movement?)
            Ahn Jung-geun is still respected in Korea today as a symbol of patriotism and courage. (How is Ahn Jung-geun perceived in Korea today?)

            Context: {context}
            """,
        )
    ]
)

# 학습 카드 포매터 Prompt
learning_cards_formatter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a formatting assistant.

            Ensure that the following learning cards are properly formatted as a JSON object with a key "learning_cards" that contains an array of objects.
            Each object should have two fields: "statement" and "question".

            Do NOT include any additional text or explanations outside the JSON object.

            Example Input:
            Ahn Jung-geun assassinated Japanese Resident General Ito Hirobumi at Harbin Station in 1909. (Who did Ahn Jung-geun assassinate at Harbin Station in 1909?)
            He organized the Korean Independence Army and fought against Japanese imperialism. (What did Ahn Jung-geun organize to fight against Japanese imperialism?)

            Example Output:
            ```json
            {{
                "learning_cards": [
                    {{
                        "statement": "Ahn Jung-geun assassinated Japanese Resident General Ito Hirobumi at Harbin Station in 1909.",
                        "question": "Who did Ahn Jung-geun assassinate at Harbin Station in 1909?"
                    }},
                    {{
                        "statement": "He organized the Korean Independence Army and fought against Japanese imperialism.",
                        "question": "What did Ahn Jung-geun organize to fight against Japanese imperialism?"
                    }}
                ]
            }}
            ```
            
            Your turn!

            Learning Cards: {context}
            """,
        )
    ]
)


def get_section_file_path(section_id: str) -> Optional[str]:
    """
    섹션 ID로 시작하는 .pdf, .docx 또는 .txt 파일을 찾기
    """
    base_dir = "/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/"
    file_pattern_pdf = f"{base_dir}{section_id}*.pdf"
    file_pattern_docx = f"{base_dir}{section_id}*.docx"
    file_pattern_txt = f"{base_dir}{section_id}*.txt"

    # 각각의 패턴으로 파일을 찾기
    matching_files = (
        glob.glob(file_pattern_pdf)
        + glob.glob(file_pattern_docx)
        + glob.glob(file_pattern_txt)
    )

    # 파일이 없는 경우 None을 반환
    if not matching_files:
        return None

    # 첫 번째 매칭되는 파일 반환 (여러 개가 있을 경우 첫 번째만 사용)
    return matching_files[0]


def create_embedding(directory_path: str, file_path: str):
    """
    임베딩을 생성하고 저장
    """
    # 파일을 읽고 문서로 변환
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    # FAISS 벡터스토어에 임베딩 저장
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(directory_path)


@router.post("/studies", response_model=StudyResponse)
async def generate_study_cards(request: StudyRequest):
    """
    /studies 엔드포인트는 사용자가 제공한 section_id를 기반으로 5개의 학습 카드를 생성하여 반환합니다.
    """
    section_id = request.section_id
    focus_prompt = request.focus_prompt or ""

    # 섹션 ID에 해당하는 벡터 스토어 폴더 경로
    embeddings_dir = f"/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/{section_id}"

    # 섹션 ID 폴더가 존재하는지 확인
    if not os.path.exists(embeddings_dir):
        raise HTTPException(
            status_code=404,
            detail=f"섹션 ID {section_id}에 해당하는 임베딩 폴더를 찾을 수 없습니다.",
        )

    try:
        # FAISS 벡터 스토어 불러오기
        vectorstore = FAISS.load_local(embeddings_dir, OpenAIEmbeddings())

        # 모든 문서를 가져오기 위해 대규모 검색 수행
        docs = vectorstore.similarity_search(
            "Retrieve all content from the document.", k=100
        )

        # 문서 내용을 하나로 병합
        context = "\n\n".join([doc.page_content for doc in docs])

        # 추가 프롬프트가 있는 경우 텍스트에 추가
        if focus_prompt:
            context += f"\n\nAdditional context: {focus_prompt}"

        # GPT 모델 설정 및 학습 카드 생성
        llm = ChatOpenAI(temperature=0.7)

        # 학습 카드 생성 체인
        learning_cards_chain = LLMChain(
            llm=llm,
            prompt=learning_cards_prompt,
        )

        # 학습 카드 생성
        learning_cards_output = learning_cards_chain.run({"context": context})
        logging.info(f"Learning Cards Output:\n{learning_cards_output}")

        # 학습 카드 포매터 체인
        formatter_chain = LLMChain(
            llm=llm,
            prompt=learning_cards_formatter_prompt,
        )

        # 학습 카드 포매팅
        formatted_learning_cards = formatter_chain.run(
            {"context": learning_cards_output}
        )
        logging.info(f"Formatted Learning Cards:\n{formatted_learning_cards}")

        print("============================================")
        # JSON 파싱
        print(formatted_learning_cards.replace("```", "")
            .replace("json", "")
            .rstrip()
            .lstrip())
        print("============================================")

        # 퀴즈 JSON 포맷으로 변환
        learning_cards_json = json.loads(
            formatted_learning_cards.replace("```", "")
            .replace("json", "")
            .rstrip()
            .lstrip()
        )

        print("---------?@#@$%#$%$%#$%#$%#$%#$%#$%#$%#$%")

        # 학습 카드 리스트 추출
        learning_cards = learning_cards_json.get("learning_cards", [])

        # 상위 5개 카드만 선택
        learning_cards = learning_cards[:5]

        return {"learning_cards": learning_cards}

    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        logging.error(f"학습 카드 생성 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"학습 카드 생성 중 오류 발생: {str(e)}"
        )
