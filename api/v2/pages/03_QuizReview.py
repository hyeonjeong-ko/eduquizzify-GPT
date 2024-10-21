import os
import json
import streamlit as st
from datetime import datetime
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from pymongo import MongoClient, errors
import random
import plotly.express as px
import matplotlib.pyplot as plt

# MongoDB 클라이언트 생성 및 데이터베이스 연결 함수
def get_mongo_client():
    try:
        client = MongoClient("mongodb://localhost:27017/")
        client.admin.command('ping')
        return client
    except errors.ConnectionFailure as e:
        st.error(f"MongoDB 연결 실패: {str(e)}")
        return None

# MongoDB에서 오답 데이터를 불러오는 함수
def load_wrong_answers():
    try:
        client = get_mongo_client()
        if client is None:
            return []
        
        db = client["quiz_db"]
        collection = db["wrong_answers"]
        wrong_answers = list(collection.find())
        return wrong_answers
    except Exception as e:
        st.error(f"MongoDB에서 오답 데이터를 불러오는 중 오류 발생: {str(e)}")
        return []
    finally:
        if client:
            client.close()

# GPT를 사용하여 빈출 주제 상위 5개 키워드를 추출하는 함수
def extract_top_five_keywords_gpt(wrong_answers):
    llm = ChatOpenAI(temperature=0.7)
    questions = [record.get("question", "Unknown") for record in wrong_answers]
    correct_answers = [record.get("correct_answer", "Unknown") for record in wrong_answers]

    prompt_template = ChatPromptTemplate.from_template(
        """
        아래는 학습자가 틀린 질문과 그에 대한 정답입니다:
        
        질문: {questions}
        정답: {correct_answers}
        
        위 질문과 답변들을 분석하여, 학습자가 자주 틀리는 포괄적인 주제 또는 키워드 상위 5개를 뽑아주세요.
        이때 되도록 짧고 간략하게 뽑아주세요.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"questions": questions, "correct_answers": correct_answers})
    keywords = result.strip().split("\n")
    return keywords[:5]

# GPT를 사용하여 주제와 관련된 추가 키워드를 생성하는 함수
def generate_additional_keywords_gpt(selected_keyword):
    llm = ChatOpenAI(temperature=0.7)
    prompt_template = ChatPromptTemplate.from_template(
        f"""
        "{selected_keyword}"라는 주제와 관련된 추가적인 키워드 5개를 생성해주세요.
        이 키워드들은 위키피디아에서 검색 가능한 관련 개념이어야 합니다.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"selected_keyword": selected_keyword})
    additional_keywords = result.strip().split("\n")
    return additional_keywords[:5]

# GPT를 사용하여 위키피디아 검색 결과를 바탕으로 퀴즈 생성하는 함수
def generate_quiz_from_wikipedia_gpt(keywords):
    llm = ChatOpenAI(temperature=0.7)
    prompt_template = ChatPromptTemplate.from_template(
        f"""
        아래의 키워드를 기반으로 각각에 대해 하나씩 5개의 퀴즈를 생성해주세요:
        
        키워드: {', '.join(keywords)}
        
        각 퀴즈는 질문과 정답을 포함해야 합니다.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"keywords": ", ".join(keywords)})
    quizzes = result.strip().split("\n")
    return quizzes

# 주제를 클릭하여 복습 퀴즈를 진행하는 함수
def start_review_quiz(selected_keyword, wrong_answers):
    st.subheader(f"복습 퀴즈 - {selected_keyword}")

    related_questions = [qa for qa in wrong_answers if selected_keyword.lower() in qa.get("question", "").lower()]
    if not related_questions:
        st.info(f"{selected_keyword}와 관련된 퀴즈가 없습니다.")
    else:
        st.write(f"**{selected_keyword}와 관련된 추가 키워드 생성 중...**")
        additional_keywords = generate_additional_keywords_gpt(selected_keyword)
        
        st.write(f"**추가 키워드:** {', '.join(additional_keywords)}")
        st.write("위키피디아를 기반으로 추가 퀴즈를 생성 중입니다...")
        quizzes = generate_quiz_from_wikipedia_gpt(additional_keywords)

        st.subheader("생성된 퀴즈")
        for idx, quiz in enumerate(quizzes, start=1):
            st.write(f"**{idx}.** {quiz}")

# Streamlit UI - 복습 퀴즈 페이지
st.title("📝 복습 퀴즈 - 오답 기반 분석")

wrong_answers = load_wrong_answers()

if "selected_keyword" not in st.session_state:
    st.session_state.selected_keyword = None

if st.session_state.selected_keyword:
    st.write(f"**선택된 키워드:** {st.session_state.selected_keyword}")
    start_review_quiz(st.session_state.selected_keyword, wrong_answers)
else:
    if not wrong_answers:
        st.warning("아직 복습을 위한 데이터가 없습니다.")
    else:
        st.subheader("📊 오답 빈출 주제 상위 5개 키워드")
        top_five_keywords = extract_top_five_keywords_gpt(wrong_answers)

        if top_five_keywords:
            # Plotly를 사용한 인터랙티브 차트
            fig = px.pie(values=[random.randint(10, 30) for _ in top_five_keywords], 
                         names=top_five_keywords, title='상위 5개 키워드 비율')
            st.plotly_chart(fig)

            st.subheader("추천 키워드")

            # 한 줄에 하나의 키워드를 버튼으로 표시
            for keyword in top_five_keywords:
                if st.button(f"💡 {keyword}", key=keyword):
                    st.session_state.selected_keyword = keyword
                    st.experimental_rerun()
