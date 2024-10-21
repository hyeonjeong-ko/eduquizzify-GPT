import os
import json
import streamlit as st
from datetime import datetime
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain, MapReduceDocumentsChain



from prompts import (
    questions_prompt_with_explanation,
    formatting_prompt_with_explanation,
    short_question_prompt,
    short_formatting_prompt,
    learning_cards_prompt,
    learning_cards_formatter_prompt,
)

from pymongo import MongoClient

# youtube
import os
import json
import streamlit as st
import subprocess
from datetime import datetime
import math
from pydub import AudioSegment
import yt_dlp
import glob
import openai


# 웹문서 링크
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st
import re
from langchain.schema import Document


# 페이지 설정
st.set_page_config(page_title="Card Quiz", page_icon="📝", layout="centered")


# MongoDB 클라이언트 생성 및 데이터베이스 연결 함수
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")  # MongoDB 서버의 URL
    return client


# MongoDB 클라이언트 생성 및 데이터베이스 연결 함수
# MongoDB에 현재 퀴즈 질문만 저장하는 함수
# MongoDB에 퀴즈 카드 저장 함수
# MongoDB에 퀴즈 카드 저장 함수
def save_quiz_to_mongo(card_data, card_type, card_idx):
    try:
        client = get_mongo_client()
        db = client["quiz_db"]

        if card_type == "quiz":
            collection = db["quizzes"]
            question = card_data["questions"][card_idx]

            # 객관식 퀴즈 저장
            if "answers" in question:  # 객관식 퀴즈
                card = {
                    "quiz_type": "multiple",  # 객관식 퀴즈
                    "question": question["question"],
                    "answers": question.get("answers", None),
                    "explanation": question.get("explanation", None),
                    "timestamp": datetime.now(),
                }
            else:  # 주관식 퀴즈
                card = {
                    "quiz_type": "short",  # 주관식 퀴즈
                    "question": question["question"],
                    "answer": question.get("answer", None),  # 주관식 답변 필드
                    "explanation": question.get("explanation", None),
                    "timestamp": datetime.now(),
                }

        elif card_type == "learning":
            collection = db["learning_cards"]
            card = card_data["learning_cards"][card_idx]
            card["timestamp"] = datetime.now()

        # 카드 데이터를 MongoDB에 저장
        collection.insert_one(card)
        st.success(f"Card {card_idx + 1}이(가) MongoDB에 저장되었습니다!")

    except Exception as e:
        st.error(f"MongoDB에 카드 저장 중 오류 발생: {str(e)}")

    finally:
        client.close()


# 오답 기록을 MongoDB에 저장하는 함수
def save_wrong_answer_to_mongo(question, correct_answer, user_answer, explanation):
    try:
        client = get_mongo_client()
        db = client["quiz_db"]  # 'quiz_db' 데이터베이스 사용
        collection = db["wrong_answers"]  # 'wrong_answers' 컬렉션 사용

        # 오답 기록 데이터
        wrong_answer_record = {
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "explanation": explanation,
            "timestamp": datetime.now(),  # 오답 발생 시간
        }
        print("##################")
        print(wrong_answer_record)

        # MongoDB에 오답 기록 저장
        collection.insert_one(wrong_answer_record)
        # st.success("오답이 MongoDB에 기록되었습니다.")

    except Exception as e:
        st.error(f"MongoDB에 오답 저장 중 오류 발생: {str(e)}")

    finally:
        client.close()


def generate_quiz_from_embeddings(
    embeddings_dir, focus_prompt="", card_type="quiz", question_type="multiple"
):
    try:
        # 임베딩 로드
        vectorstore = FAISS.load_local(
            embeddings_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(
            "Retrieve all content from the document.", k=100
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        context += f"\n\nAdditional context: {focus_prompt}"

        # 카드 유형에 맞는 프롬프트 설정
        llm = ChatOpenAI(temperature=0.7)

        if card_type == "quiz":  # 퀴즈 카드 생성
            if question_type == "multiple":
                chain = LLMChain(llm=llm, prompt=questions_prompt_with_explanation)
                formatting_chain = LLMChain(
                    llm=llm, prompt=formatting_prompt_with_explanation
                )
            elif question_type == "short":
                chain = LLMChain(llm=llm, prompt=short_question_prompt)
                formatting_chain = LLMChain(llm=llm, prompt=short_formatting_prompt)
            else:
                raise ValueError("Invalid question type selected")
        elif card_type == "learning":  # 학습 카드 생성
            chain = LLMChain(llm=llm, prompt=learning_cards_prompt)
            formatting_chain = LLMChain(llm=llm, prompt=learning_cards_formatter_prompt)
        else:
            raise ValueError(f"Invalid card type selected: {card_type}")

        # 카드 생성 및 포매팅
        card_output = chain.run({"context": context})
        formatted_card = formatting_chain.run({"context": card_output})
        card_data = json.loads(formatted_card.replace("```", "").replace("json", ""))

        return card_data

    except Exception as e:
        st.error(f"Error generating cards: {str(e)}")
        return None


# 새로운 퀴즈 생성 모달 함수 정의
# 새로운 퀴즈 생성 모달 함수 정의
@st.dialog("마지막 질문입니다. 퀴즈를 더 생성하시겠습니까?")
def quiz_more_prompt():
    st.write("추가 퀴즈를 생성하시겠습니까?")

    # 버튼 스타일 추가
    button_style = """
        <style>
        .modal-button {
            width: 80px;  /* 버튼 너비 조정 */
            background-color: #4CAF50;
            color: white;
            padding: 6px 10px;
            border-radius: 5px;
            border: none;
            font-size: 13px;
            cursor: pointer;
        }
        .modal-button:hover {
            background-color: #45a049;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("예", key="yes_button", help="추가 퀴즈를 생성합니다."):
            # 퀴즈 재생성 로직 초기화
            st.session_state.current_card = 0  # 현재 카드 인덱스 초기화
            st.session_state.quiz_data = None  # 기존 퀴즈 데이터 초기화

            # 페이지를 다시 퀴즈 생성으로 설정
            st.query_params = {"page": "card_generation"}
            st.rerun()  # 페이지를 새로고침하여 퀴즈 생성 로직을 다시 실행

    with col2:
        if st.button("아니오", key="no_button", help="홈 페이지로 돌아갑니다."):
            # 아니오를 선택하면 세션을 초기화하고 홈으로 이동
            st.session_state.clear()  # 세션 상태 초기화
            st.query_params = {"page": "quiz"}  # 첫 페이지로 이동 설정
            st.rerun()


# 퀴즈 생성 함수 정의
# 임베딩과 카드를 생성하는 함수 (퀴즈 카드와 학습 카드를 처리)
def generate_cards_from_embeddings(
    embeddings_dir, focus_prompt="", card_type="quiz", question_type="multiple"
):
    try:
        # 임베딩 로드
        vectorstore = FAISS.load_local(embeddings_dir, OpenAIEmbeddings())
        docs = vectorstore.similarity_search(
            "Retrieve all content from the document.", k=100
        )
        context = "\n\n".join([doc.page_content for doc in docs])
        context += f"\n\nAdditional context: {focus_prompt}"

        # 카드 유형에 맞는 프롬프트 설정
        llm = ChatOpenAI(temperature=0.7)

        if card_type == "quiz":  # 퀴즈 카드 생성
            if question_type == "multiple":
                chain = LLMChain(llm=llm, prompt=questions_prompt_with_explanation)
                formatting_chain = LLMChain(
                    llm=llm, prompt=formatting_prompt_with_explanation
                )
            elif question_type == "short":
                chain = LLMChain(llm=llm, prompt=short_question_prompt)
                formatting_chain = LLMChain(llm=llm, prompt=short_formatting_prompt)
            else:
                raise ValueError("Invalid question type selected")
        elif card_type == "learning":  # 학습 카드 생성
            chain = LLMChain(llm=llm, prompt=learning_cards_prompt)
            formatting_chain = LLMChain(llm=llm, prompt=learning_cards_formatter_prompt)
        else:
            raise ValueError(f"Invalid card type selected: {card_type}")

        # 카드 생성 및 포매팅
        card_output = chain.run({"context": context})
        formatted_card = formatting_chain.run({"context": card_output})
        card_data = json.loads(formatted_card.replace("```", "").replace("json", ""))

        return card_data

    except Exception as e:
        st.error(f"Error generating cards: {str(e)}")
        return None


# 파일 저장 경로 생성 함수
def save_uploaded_file(uploaded_file):
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = (
        f"./.cache/file_embeddings/{current_date}_{uploaded_file.name.split('.')[0]}"
    )
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, save_dir


# 임베딩 생성 및 저장 함수
def create_embedding(file_path, save_dir):
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_dir)

    return f"{save_dir}/index.faiss", f"{save_dir}/index.pkl"


# 현재 퀴즈의 인덱스를 관리
def show_quiz_card(quiz_data, question_idx):
    question = quiz_data["questions"][question_idx]
    st.subheader(f"Question {question_idx + 1}: {question['question']}")
    user_answer = st.radio(
        "Select your answer:", [a["answer"] for a in question["answers"]]
    )

    if st.button("Submit Answer"):
        correct_answer = next(a["answer"] for a in question["answers"] if a["correct"])
        if user_answer == correct_answer:
            st.success(f"정답입니다! {correct_answer}")
        else:
            st.error(f"오답입니다. 정답은 {correct_answer}입니다.")

            st.write("save_wrong_answer실행!!!!!!!!!!!")
            # 사용자가 틀린 문제를 MongoDB에 저장
            save_wrong_answer_to_mongo(
                question["question"],
                correct_answer,
                user_answer,
                question["explanation"],
            )

        st.write(f"**Explanation**: {question['explanation']}")


def show_short_answer_card(quiz_data, question_idx):
    question = quiz_data["questions"][question_idx]
    st.subheader(f"Question {question_idx + 1}: {question['question']}")
    user_answer_key = f"user_answer_{question_idx}"
    user_answer = st.text_input("Enter your answer:", key=user_answer_key)

    if st.button("Submit Answer"):
        st.success(f"Correct Answer: {question['answer']}")


# 페이지 상태 관리
# 학습 카드 또는 퀴즈 카드를 화면에 표시하는 함수
# 카드 보여주기 함수
def show_card(card_data, card_type, card_idx):
    if card_type == "quiz":
        question = card_data["questions"][card_idx]
        st.subheader(f"Card {card_idx + 1}")
        st.write(f"**Question**: {question['question']}")

        if "answers" in question:
            user_answer = st.radio(
                "Select your answer:", [a["answer"] for a in question["answers"]]
            )
            if st.button("Submit Answer"):
                correct_answer = next(
                    a["answer"] for a in question["answers"] if a["correct"]
                )
                if user_answer == correct_answer:
                    st.success(f"정답입니다! {correct_answer}")
                else:
                    st.error(f"오답입니다. 정답은 {correct_answer}입니다.")
                    # 사용자가 틀린 문제를 MongoDB에 저장
                    save_wrong_answer_to_mongo(
                        question["question"],
                        correct_answer,
                        user_answer,
                        question["explanation"],
                    )

                st.write(
                    f"**Explanation**: {question.get('explanation', 'No explanation available.')}"
                )
        else:
            user_answer = st.text_input("Enter your answer:")
            if st.button("Submit Answer"):
                st.success(f"Correct Answer: {question['answer']}")

    elif card_type == "learning":
        card = card_data["learning_cards"][card_idx]
        st.subheader(f"Card {card_idx + 1}")
        st.write(f"**Statement**: {card['statement']}")
        st.write(f"**Question**: {card['question']}")
        if "explanation" in card:
            st.write(f"**Explanation**: {card['explanation']}")


def preprocess_blog_content(raw_content):
    """
    주어진 블로그 텍스트에서 불필요한 정보(메뉴, 카테고리, 댓글, 통계 등)를 제거하고
    본문만 남기는 전처리 함수.

    Parameters:
    raw_content (list of Document or str): 블로그의 원시 텍스트 리스트나 단일 텍스트

    Returns:
    str: 전처리된 본문 텍스트
    """
    # 만약 raw_content가 Document 객체들의 리스트라면 각 page_content를 추출하여 결합
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            [doc.page_content for doc in raw_content if isinstance(doc, Document)]
        )

    # 1. HTML 태그 제거
    cleaned_content = re.sub(r"<.*?>", "", raw_content)

    # 2. 불필요한 메뉴/카테고리/태그 관련 정보 제거
    patterns_to_remove = [
        r"바로가기.*?닫기",  # 메뉴 관련 텍스트 제거
        r"CATEGORY.*?(\n\n|\\n\\n)",  # 카테고리 및 하위 메뉴 제거
        r"TAG.*?(\n\n|\\n\\n)",  # 태그 제거
        r"공유하기.*?\n\n",  # 공유하기 관련 정보 제거
        r"최근에 올라온 글.*?\n\n",  # 최근 글 리스트 제거
        r"최근에 달린 댓글.*?\n\n",  # 최근 댓글 리스트 제거
        r"댓글쓰기 폼.*?\n\n",  # 댓글 폼 제거
        r"Powered by.*?Tistory",  # Tistory 관련 푸터 정보 제거
        r"\n{2,}",  # 연속적인 빈 줄 제거
        r"\s*반응형\s*",  # '반응형' 텍스트 제거
        r"^\s*$",  # 빈 줄 제거
        r"단축키.*?\n\n",  # 단축키 안내 제거
    ]

    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)

    # 3. 기타 불필요한 공백 정리
    cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()

    return cleaned_content


# MapReduce 요약 생성 함수
# MapReduce 요약 생성 함수
def summarize_youtube_script_with_map_reduce(transcript_path, user_request):
    # 스크립트를 청크로 분할
    loader = UnstructuredFileLoader(transcript_path)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    # 요약을 위한 프롬프트 정의 (Map 단계)
    map_prompt_template = """
    Summarize the following transcript chunk, focusing on the user's request: {user_request}
    Text: {input_documents}
    """
    map_prompt = PromptTemplate.from_template(map_prompt_template)

    # Map 단계의 LLMChain 정의
    map_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.7), prompt=map_prompt
    )

    # 요약을 결합하기 위한 Reduce 단계의 프롬프트 정의
    reduce_prompt_template = """
    Combine the following summaries into a concise summary, focusing on the user's request: {user_request}
    Summaries: {input_documents}
    """
    reduce_prompt = PromptTemplate.from_template(reduce_prompt_template)

    # Reduce 단계의 LLMChain 정의
    reduce_llm_chain = LLMChain(
        llm=ChatOpenAI(temperature=0.7), prompt=reduce_prompt
    )

    # Reduce 단계를 위해 StuffDocumentsChain 사용
    reduce_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain, document_variable_name="input_documents"
    )

    # MapReduce 문서 체인 구성
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,  # Map 단계
        reduce_documents_chain=reduce_documents_chain,  # Reduce 단계
        document_variable_name="input_documents"  # 청크 처리
    )

    # 최종 요약을 생성
    final_summary = map_reduce_chain.run({
        "input_documents": docs,  # 문서 청크
        "user_request": user_request  # 사용자 요청 반영
    })

    return final_summary


# 페이지 상태 관리
query_params = st.query_params

# 페이지 관리
if "page" in query_params:
    page = query_params["page"]
else:
    page = "quiz"

# 파일 업로드 상태 관리
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# 카드 데이터 세션 상태 초기화
if "card_data" not in st.session_state:
    st.session_state.card_data = None

if "current_card" not in st.session_state:
    st.session_state.current_card = 0

# 페이지에 따른 화면 표시
if page == "quiz":
    st.title("자료 유형을 선택하세요")
    options = [
        "공식 docs 링크",
        "웹문서 자료 링크",
        "개인 파일 업로드 (PDF, TXT)",
        "유튜브 영상 링크",
    ]

    selection = st.radio("", options)
    if st.button("Next"):
        if selection == "개인 파일 업로드 (PDF, TXT)":
            st.query_params = {"page": "file_upload"}  # 쿼리 파라미터 설정
        elif selection == "유튜브 영상 링크":
            st.query_params = {
                "page": "youtube_input"
            }  # 유튜브 링크 입력 페이지로 이동
        elif selection == "웹문서 자료 링크":
            st.query_params = {"page": "web_input"}

# 유튜브 영상 링크일때#####################


# 유튜브 영상 다운로드 및 mp3로 추출 함수
def download_audio_with_ytdlp(youtube_link, output_path):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",  # 영상명에 따라 저장
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            youtube_link, download=True
        )  # 영상 정보와 함께 다운로드
        return info["title"]  # 다운로드된 영상의 제목 반환


# mp3를 10분 단위로 분할하여 청크로 저장하는 함수
def split_audio_into_chunks(audio_path, chunk_dir, chunk_duration=10):
    track = AudioSegment.from_mp3(audio_path)
    ten_minutes = chunk_duration * 60 * 1000  # 10분 단위
    chunks = math.ceil(len(track) / ten_minutes)

    os.makedirs(chunk_dir, exist_ok=True)  # 청크 디렉토리 생성

    for i in range(chunks):
        start_time = i * ten_minutes
        end_time = (i + 1) * ten_minutes

        chunk = track[start_time:end_time]
        chunk.export(f"{chunk_dir}/chunk_{i}.mp3", format="mp3")


# OpenAI Whisper API를 사용하여 각 청크 파일의 자막을 추출하는 함수
def transcribe_chunks(chunk_folder, destination):
    # 청크 파일들을 glob을 통해 가져옴
    files = glob.glob(f"{chunk_folder}/*.mp3")

    # 파일 순서대로 처리
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # OpenAI API를 사용해 자막 생성
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            # 자막을 파일에 저장
            text_file.write(transcript["text"] + "\n")


# Refine 요약 생성
def refine_summary(transcript_path, user_request):
    loader = UnstructuredFileLoader(transcript_path)
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    llm = ChatOpenAI(temperature=0.1)
    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following, considering the user's request: "{user_request}"
        "{text}"
        CONCISE SUMMARY:                
    """
    )
    first_summary_chain = LLMChain(llm=llm, prompt=first_summary_prompt)
    summary = first_summary_chain.run(
        {"text": docs[0].page_content, "user_request": user_request}
    )

    refine_prompt = ChatPromptTemplate.from_template(
        """
        Refine the existing summary using the following context:
        "{context}"
        Considering the user's request: "{user_request}"
        REFINED SUMMARY:                
    """
    )
    refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

    for doc in docs[1:]:
        summary = refine_chain.run(
            {
                "existing_summary": summary,
                "context": doc.page_content,
                "user_request": user_request,
            }
        )

    return summary


# 유튜브 영상 처리 페이지

# 웹 자료 입력 페이지
if page == "web_input":
    st.title("웹 URL을 입력하세요")
    web_url = st.text_input("웹 자료 링크", placeholder="https://example.com")

    if st.button("Submit"):
        if web_url:
            st.session_state["web_url"] = web_url
            st.success("웹 링크가 성공적으로 저장되었습니다!")
            st.query_params = {"page": "web_processing"}
            st.rerun()
        else:
            st.error("웹 링크를 입력하세요.")

# 웹 자료 처리 페이지
# 웹 자료 처리 페이지
if page == "web_processing":
    st.title("웹 콘텐츠를 가져오는 중...")
    web_url = st.session_state["web_url"]

    with st.spinner("웹 페이지에서 콘텐츠를 가져오는 중입니다..."):
        loader = AsyncChromiumLoader([web_url])
        docs = loader.load()
        html2text_transformer = Html2TextTransformer()
        transformed_docs = html2text_transformer.transform_documents(docs)

    # 웹 콘텐츠 전처리
    cleaned_content = preprocess_blog_content(transformed_docs)
    # st.write(cleaned_content)

    st.session_state["web_content"] = (
        cleaned_content  # 전처리된 웹 콘텐츠를 세션에 저장
    )
    st.success("웹 콘텐츠가 성공적으로 처리되었습니다!")

    # 추가 요청 사항 입력 및 카드 선택
    st.title("퀴즈 또는 학습 카드 설정")

    additional_request = st.text_area(
        "추가 요청 사항 입력", placeholder="예: 특정 주제에 집중"
    )

    # 카드 유형 선택 (퀴즈 카드 or 학습 카드)
    card_type_selection = st.radio("카드 유형 선택:", ("퀴즈 카드", "학습 카드"))
    card_type = "quiz" if card_type_selection == "퀴즈 카드" else "learning"

    # 질문 유형 선택 (퀴즈 카드일 경우만)
    if card_type == "quiz":
        question_type_selection = st.radio("질문 유형 선택:", ("객관식", "주관식"))
        question_type = "multiple" if question_type_selection == "객관식" else "short"
    else:
        question_type = None  # 학습 카드일 경우 질문 유형은 필요 없음

    # 카드 생성 버튼
    if st.button("카드 생성"):
        st.session_state["additional_request"] = additional_request
        st.session_state["card_type"] = card_type
        st.session_state["question_type"] = question_type
        st.query_params = {"page": "web_card_generation"}  # 카드 생성 페이지로 이동
        st.rerun()

# 웹 기반 카드 생성 페이지
if page == "web_card_generation":
    st.title("카드 생성 중...")

    # 세션에서 웹 콘텐츠와 추가 요청 사항 불러오기
    web_content = st.session_state.get("web_content", "")
    additional_request = st.session_state.get("additional_request", "")

    # 카드 유형과 질문 유형에 따라 임베딩 생성 및 카드 생성
    with st.spinner("카드를 생성하는 중입니다..."):
        try:
            # 임베딩 생성에 필요한 정보
            embeddings_dir = (
                f"./.cache/web_embeddings/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(embeddings_dir, exist_ok=True)

            # 웹 콘텐츠를 기반으로 임베딩 생성
            summary_file_path = f"{embeddings_dir}/web_content.txt"
            with open(summary_file_path, "w") as f:
                f.write(web_content)

            faiss_path, pkl_path = create_embedding(summary_file_path, embeddings_dir)

            # 세션에 임베딩 경로 저장
            st.session_state["faiss_path"] = faiss_path
            st.session_state["pkl_path"] = pkl_path

            # 카드 생성 호출
            card_data = generate_quiz_from_embeddings(
                embeddings_dir,
                additional_request,
                st.session_state["card_type"],
                st.session_state["question_type"],
            )

            if card_data:
                st.success("카드 생성 완료!")
                st.session_state.card_data = (
                    card_data  # 생성된 카드 데이터를 세션에 저장
                )
                st.query_params = {"page": "card_display"}  # 카드 표시 페이지로 이동
                st.rerun()
            else:
                st.error("카드 생성 실패!")
        except Exception as e:
            st.error(f"임베딩 생성 및 카드 생성 중 오류 발생: {str(e)}")

# 유튜브 영상 처리 페이지
if page == "youtube_input":
    st.title("유튜브 영상 링크를 입력하세요")
    youtube_link = st.text_input(
        "유튜브 영상 링크", placeholder="유튜브 URL을 입력하세요"
    )

    if st.button("Submit"):
        if youtube_link:
            st.session_state["youtube_link"] = youtube_link
            st.success("유튜브 링크가 성공적으로 저장되었습니다!")
            st.query_params = {"page": "quiz_settings"}
            st.rerun()
        else:
            st.error("유튜브 링크를 입력하세요.")


# 퀴즈 설정 페이지
if page == "quiz_settings":
    st.title("퀴즈 설정")

    additional_request = st.text_area(
        "퀴즈 생성에 대한 추가 요청 사항 입력", placeholder="예: 특정 주제에 집중"
    )

    card_type_selection = st.radio("카드 유형 선택:", ("퀴즈 카드", "학습 카드"))
    card_type = "quiz" if card_type_selection == "퀴즈 카드" else "learning"

    if card_type == "quiz":
        question_type_selection = st.radio("질문 유형 선택:", ("객관식", "주관식"))
        question_type = "multiple" if question_type_selection == "객관식" else "short"
    else:
        question_type = None

    if st.button("카드 생성"):
        st.session_state["additional_request"] = additional_request
        st.session_state["card_type"] = card_type
        st.session_state["question_type"] = question_type
        st.query_params = {"page": "youtube_processing"}
        st.rerun()


# 유튜브 영상 처리 및 카드 생성
if page == "youtube_processing":
    st.title("유튜브 영상 처리 중...")

    youtube_link = st.session_state["youtube_link"]
    additional_request = st.session_state["additional_request"]

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"./files/youtube/{current_date}"
    os.makedirs(video_dir, exist_ok=True)

    with st.spinner("유튜브에서 오디오를 다운로드 중입니다..."):
        video_title = download_audio_with_ytdlp(youtube_link, video_dir)
        audio_path = f"{video_dir}/{video_title}.mp3"

    with st.spinner("오디오를 10분 단위로 청크 분할 중입니다..."):
        chunk_dir = f"{video_dir}/chunks"
        split_audio_into_chunks(audio_path, chunk_dir)

    transcription_file = f"{video_dir}/transcription.txt"
    with st.spinner("OpenAI Whisper로 자막을 생성 중입니다..."):
        transcribe_chunks(chunk_dir, transcription_file)

    # with st.spinner("요약을 생성 중입니다..."):
    #     summary = refine_summary(transcription_file, additional_request)
    #     st.session_state["summary"] = summary

    with st.spinner("MapReduce로 요약을 생성 중입니다..."):
        summary = summarize_youtube_script_with_map_reduce(
            transcription_file, additional_request
        )
        st.session_state["summary"] = summary

    embeddings_dir = f"./.cache/youtube_embeddings/{current_date}"
    os.makedirs(embeddings_dir, exist_ok=True)

    with st.spinner("요약된 내용을 기반으로 임베딩을 생성 중입니다..."):
        try:
            summary_file_path = f"{video_dir}/summary.txt"
            with open(summary_file_path, "w") as f:
                f.write(summary)

            faiss_path, pkl_path = create_embedding(summary_file_path, embeddings_dir)

            st.session_state["faiss_path"] = faiss_path
            st.session_state["pkl_path"] = pkl_path
            st.session_state["embeddings_dir"] = embeddings_dir

        except Exception as e:
            st.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            st.stop()

    st.success(f"요약 및 임베딩 생성 완료! 이제 카드를 생성합니다.")
    st.query_params = {"page": "card_generation"}
    st.rerun()

###################################################################


elif page == "file_upload":
    st.title("자료를 업로드하세요")

    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 또는 TXT 파일 업로드", type=["pdf", "txt"])
    if uploaded_file is not None:
        st.session_state["uploaded_file_path"], save_dir = save_uploaded_file(
            uploaded_file
        )
        st.write(f"파일 업로드 완료: {uploaded_file.name}")

        additional_request = st.text_area(
            "작업 추가 요청 내용 입력 (선택 사항)",
            placeholder="예: 특정 핵심 내용만, 한국어로만 답변해주세요 등",
        )

        card_type_selection = st.radio("카드 유형 선택:", ("퀴즈 카드", "학습 카드"))
        card_type = "quiz" if card_type_selection == "퀴즈 카드" else "learning"

        # 질문 유형 (퀴즈 카드일 경우만 선택)
        if card_type == "quiz":
            question_type_selection = st.radio("질문 유형 선택:", ("객관식", "주관식"))

            # Map the selection to internal types
            if question_type_selection == "객관식":
                question_type = "multiple"
            elif question_type_selection == "주관식":
                question_type = "short"
        else:
            question_type = None  # 학습 카드일 경우 질문 유형 필요 없음

        if st.button("임베딩 생성 및 다음 단계"):
            st.session_state["additional_request"] = additional_request
            st.session_state["card_type"] = card_type
            st.session_state["question_type"] = question_type

            # 임베딩 생성 시 로딩 스피너 추가
            with st.spinner("임베딩을 생성하고 있습니다. 잠시만 기다려주세요..."):
                faiss_path, pkl_path = create_embedding(
                    st.session_state["uploaded_file_path"], save_dir
                )

            # 임베딩 파일 경로 세션 저장
            st.session_state["faiss_path"] = faiss_path
            st.session_state["pkl_path"] = pkl_path
            st.session_state["embeddings_dir"] = save_dir  # 임베딩 디렉터리 저장

            # 카드 생성 페이지로 이동
            st.query_params = {"page": "card_generation"}  # 쿼리 파라미터 설정
            st.rerun()  # 페이지 리로드

elif page == "card_generation":
    st.title("카드 생성 중...")

    # "uploaded_file_path"가 세션 상태에 있는지 확인하고 기본값 설정
    if "uploaded_file_path" in st.session_state:
        st.write(f"업로드된 파일: {st.session_state['uploaded_file_path']}")
    else:
        st.warning("업로드된 파일이 없습니다.")

    if "faiss_path" in st.session_state:
        st.write(f"FAISS 경로: {st.session_state['faiss_path']}")
    else:
        st.warning("FAISS 경로를 찾을 수 없습니다.")

    if "pkl_path" in st.session_state:
        st.write(f"PKL 경로: {st.session_state['pkl_path']}")
    else:
        st.warning("PKL 경로를 찾을 수 없습니다.")

    st.write(
        f"선택한 카드 유형: {st.session_state.get('card_type', '카드 유형이 선택되지 않았습니다.')}"
    )
    if st.session_state.get("card_type") == "quiz":
        st.write(
            f"선택한 질문 유형: {st.session_state.get('question_type', '질문 유형이 선택되지 않았습니다.')}"
        )
    st.write(
        f"추가 요청 사항: {st.session_state.get('additional_request', '추가 요청 사항 없음')}"
    )

    # 카드 생성 호출
    if "faiss_path" in st.session_state:
        with st.spinner("카드를 생성하는 중입니다..."):
            embeddings_dir = os.path.dirname(st.session_state["faiss_path"])
            card_data = generate_quiz_from_embeddings(
                embeddings_dir,
                st.session_state["additional_request"],
                st.session_state["card_type"],
                st.session_state["question_type"],
            )

        if card_data:
            st.success("카드 생성 완료!")

            # 카드 데이터를 세션에 저장
            st.session_state.card_data = card_data
            st.query_params = {"page": "card_display"}  # 카드 표시 페이지로 이동
            st.rerun()
        else:
            st.error("카드 생성 실패!")
    else:
        st.error("FAISS 경로가 설정되지 않았습니다.")


# 카드 표시 페이지 추가
elif page == "card_display":
    st.title("카드 결과")

    card_data = st.session_state.card_data
    card_type = st.session_state.card_type

    if card_data:
        current_card_idx = st.session_state.current_card

        # 카드 보여주기 함수 호출
        show_card(card_data, card_type, current_card_idx)

        # HTML과 CSS를 사용해 버튼 스타일 추가
        button_style = """
            <style>
            .stButton button {
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 6px 12px; /* 패딩을 줄여서 더 슬림하게 만듦 */
                border-radius: 5px; /* 둥근 모서리 크기를 줄임 */
                border: none;
                font-size: 13px; /* 폰트 크기를 줄여서 슬림하게 */
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #45a049; /* 호버시 약간 어두운 색상 */
            }
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        # 버튼을 두 개 배치 (스타일링 적용)
        col1, col2, col3 = st.columns([5, 1, 1])  # 왼쪽을 넓히고 오른쪽을 좁힘

        with col2:
            if st.button("카드 저장"):
                st.write("카드 저장됨")
                save_quiz_to_mongo(card_data, card_type, current_card_idx)

        with col3:
            if st.button("다음 카드"):
                if card_type == "quiz":
                    max_index = len(card_data["questions"]) - 1
                else:
                    max_index = len(card_data["learning_cards"]) - 1

                if current_card_idx < max_index:
                    st.session_state.current_card += 1
                    st.rerun()  # 다음 카드로 넘어감
                else:
                    # 마지막 카드이면 모달 띄우기
                    quiz_more_prompt()  # 모달 호출
    else:
        st.error("카드 데이터를 가져올 수 없습니다.")
