import os
import json
from typing import Optional
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

from fastapi import APIRouter

router = APIRouter()


@router.get("/test-quiz")
async def test_summarize():
    return {"message": "quiz endpoint is working!"}


# 환경 변수 로드
load_dotenv()

# app = FastAPI()


# Quiz 요청 데이터 모델 정의
class QuizRequest(BaseModel):
    section_id: str  # 섹션 ID
    quiz_type: str  # 퀴즈 유형 (객관식/주관식, 현재는 객관식만 처리)
    focus_prompt: Optional[str] = None  # 사용자가 추가한 프롬프트 (선택 사항)


class StudyRequest(BaseModel):
    sectionId: str
    focusPrompt = Optional[str] = None


# 객관식; 퀴즈 생성 및 해설 추가 Prompt
questions_prompt_with_explanation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
             
            Based ONLY on the following context make 5 (FIVE) questions to test the user's knowledge about the text.
            
            Each question should have 5 answers, four of them must be incorrect and one should be correct.
             
            After each question, provide a brief explanation to clarify the correct answer.

            Use (o) to signal the correct answer.
             
            Question examples:
             
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)|Purple
            Explanation: The ocean appears blue because water absorbs colors in the red part of the light spectrum.

            Question: What is the capital of Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut|Kyiv
            Explanation: Tbilisi is the capital of Georgia, located in the Caucasus region of Eurasia.
             
            Your turn!
             
            Context: {context}
            """,
        )
    ]
)

# JSON 포맷으로 변환 Prompt (해설 포함)
formatting_prompt_with_explanation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
             
            You format exam questions with explanations into JSON format.
            Answers with (o) are the correct ones.
             
            Example Input:

            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
            Explanation: The ocean appears blue because water absorbs colors in the red part of the light spectrum.
                 
            Question: What is the capital of Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut|Kyiv
            Explanation: Tbilisi is the capital of Georgia, located in the Caucasus region of Eurasia.
                 
            Example Output:
             
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the ocean?",
                        "answers": [
                                {{
                                    "answer": "Red",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Yellow",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Green",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Blue",
                                    "correct": true
                                }}
                        ],
                        "explanation": "The ocean appears blue because water absorbs colors in the red part of the light spectrum."
                    }},
                    {{
                        "question": "What is the capital of Georgia?",
                        "answers": [
                                {{
                                    "answer": "Baku",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Tbilisi",
                                    "correct": true
                                }},
                                {{
                                    "answer": "Manila",
                                    "correct": false
                                }},
                                {{
                                    "answer": "Beirut",
                                    "correct": false
                                }}
                        ],
                        "explanation": "Tbilisi is the capital of Georgia, located in the Caucasus region of Eurasia."
                    }}
                ]
             }}
            ```
            Your turn!

            Questions: {context}
            """,
        )
    ]
)
# Short-Answer Quiz Prompt
short_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            
            Based ONLY on the following context, generate 5 (FIVE) short-answer questions to test the user's knowledge about the text.
            
            Each question should be open-ended and require the user to provide a brief answer.
            
            Question example:
            
            Question: What is the color of the ocean?
            Answer: Blue
             
            Your turn!
             
            Context: {context}
            """,
        )
    ]
)

short_formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
             
            You format short-answer quiz questions into JSON format.
             
            Example Input:

            Question: What is the color of the ocean?
            Answer: Blue
                 
            Question: What is the capital of Georgia?
            Answer: Tbilisi
             
            Example Output:
             
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the ocean?",
                        "answer": "Blue"
                    }},
                    {{
                        "question": "What is the capital of Georgia?",
                        "answer": "Tbilisi"
                    }}
                ]
             }}
            ```
            Your turn!

            Questions: {context}
            """,
        )
    ]
)


def get_section_file_path(section_id):
    # 섹션 ID로 시작하는 .pdf, .docx 또는 .txt 파일을 찾기
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


@router.post("/quizes")
async def generate_quiz(request: QuizRequest):
    section_id = request.section_id
    focus_prompt = request.focus_prompt or ""

    # 섹션 ID에 해당하는 벡터 스토어 폴더 경로
    embeddings_dir = f"/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/{section_id}"

    # 섹션 ID에 해당하는 파일 경로 가져오기 (pdf 또는 docx)
    try:
        file_path = get_section_file_path(section_id)
        print("file_path", file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 섹션 ID 폴더가 존재하지 않으면 임베딩 생성
    if not os.path.exists(embeddings_dir):
        if file_path:
            # 파일이 존재하면 임베딩 생성
            create_embedding(embeddings_dir, file_path)
        else:
            # 파일이 없으면 적절한 파일이 없다는 메시지 반환
            raise HTTPException(
                status_code=404,
                detail=f"섹션 ID {section_id}에 해당하는 파일을 찾을 수 없고 임베딩을 생성할 수 없습니다.",
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
        context += f"\n\nAdditional context: {focus_prompt}"

        # GPT 모델 설정 및 퀴즈 생성
        llm = ChatOpenAI(temperature=0.7)

        # 퀴즈 유형에 따라 다른 Prompt 설정 (객관식/주관식)
        if request.quiz_type == "multiple":
            # 객관식
            chain = LLMChain(
                llm=llm,
                prompt=questions_prompt_with_explanation,
            )
            formatting_chain = LLMChain(
                llm=llm,
                prompt=formatting_prompt_with_explanation,
            )
        elif request.quiz_type == "short":
            # 주관식
            chain = LLMChain(llm=llm, prompt=short_question_prompt)
            formatting_chain = LLMChain(llm=llm, prompt=short_formatting_prompt)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"지원되지 않는 퀴즈 유형입니다: {request.quiz_type}",
            )

        # 퀴즈 생성
        questions_output = chain.run({"context": context})

        # JSON 포맷으로 변환
        formatted_quiz = formatting_chain.run({"context": questions_output})

        # 퀴즈 JSON 포맷으로 변환
        quiz_data = json.loads(formatted_quiz.replace("```", "").replace("json", ""))

        print("=================")
        print(quiz_data)

        return {"quiz": quiz_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")


@router.post("/studyings")
async def study_docs(request: StudyRequest):
    """
    sectionId를 받아 vectorestore캐시 기반 학습내용 생성
    """
    section_id = request.sectionId
    focus_prompt = request.focusPrompt

    # section_id에 해당하는 벡터스토어가져오기

    # 섹션 ID에 해당하는 벡터 스토어 폴더 경로
    embeddings_dir = f"/Users/gohyeonjeong/Desktop/myFolder/vscode-projects/EduQuizzify-GPT/.cache/file_embeddings/{section_id}"

    # 섹션 ID 폴더가 존재하지 않으면 Error
    if not os.path.exists(embeddings_dir):
        if not embeddings_dir:
            raise HTTPException(
                status_code=404,
                detail=f"섹션 ID {section_id}에 해당하는 파일을 찾을 수 없고 임베딩을 생성할 수 없습니다.",
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
        context += f"\n\nAdditional context: {focus_prompt}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")
