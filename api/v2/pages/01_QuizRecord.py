import streamlit as st
from pymongo import MongoClient

# MongoDB 클라이언트 생성 및 데이터베이스 연결 함수
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")  # MongoDB 서버의 URL
    return client

# 몽고DB에서 퀴즈 데이터를 불러오는 함수
def load_quizzes():
    client = get_mongo_client()
    db = client["quiz_db"]
    collection = db["quizzes"]
    quizzes = list(collection.find())  # 퀴즈 목록을 가져옴
    client.close()  # 연결 닫기
    return quizzes

# CSS for card flip effect with reduced size and text adjustment
card_flip_css = """
<style>
.flip-card {
  background-color: transparent;
  width: 220px;
  height: 180px;
  perspective: 1000px;
  margin: 15px;
}

.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s;
  transform-style: preserve-3d;
  box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}

.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
  border-radius: 15px;
  overflow: hidden; /* Prevent overflow of text */
}

.flip-card-front {
  background-color: #fff;
  color: black;
  font-size: 12px; /* Adjust font size */
  padding: 10px; /* Add padding for text */
  overflow-y: auto; /* Scrollable text */
}

.flip-card-back {
  background-color: #2980b9;
  color: white;
  transform: rotateY(180deg);
  font-size: 12px; /* Adjust font size */
  padding: 10px;
  overflow-y: auto; /* Scrollable text */
}

.flip-card:hover .flip-card-inner {
  transform: rotateY(180deg);
}

.correct {
  background-color: #d4edda;
  color: #155724;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 5px;
}

.incorrect {
  background-color: #f8d7da;
  color: #721c24;
  padding: 10px;
  border-radius: 5px;
  margin-bottom: 5px;
}

.explanation {
  background-color: #f1f3f5;
  padding: 10px;
  margin-top: 10px;
  border-left: 5px solid #2980b9;
  border-radius: 5px;
}
</style>
"""

# 주관식 퀴즈를 플립 카드로 보여주는 함수
def show_quiz_card_with_flip(quiz, idx):
    st.markdown(card_flip_css, unsafe_allow_html=True)

    # 주관식 퀴즈에서 "question"과 "answer"를 올바르게 참조
    question_text = quiz.get("question", "")  # "question"이 객체가 아닐 수도 있으므로 기본값 설정
    answer_text = quiz.get("answer", "정답이 없습니다.")  # "answer" 필드를 가져옴

    flip_card_html = f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <p>{question_text}</p>
        </div>
        <div class="flip-card-back">
          <h3>정답</h3>
          <p>{answer_text}</p>
        </div>
      </div>
    </div>
    """
    st.markdown(flip_card_html, unsafe_allow_html=True)

# 주관식 퀴즈를 3x3 그리드로 보여주는 함수
def show_short_quizzes_in_grid(short_quizzes):
    num_cards = len(short_quizzes)
    cols_per_row = 3  # 한 줄에 3개의 카드 배치
    rows = num_cards // cols_per_row + (1 if num_cards % cols_per_row else 0)  # 필요한 행 수 계산

    for row in range(rows):
        cols = st.columns(3)  # 3개의 열 생성
        for col_idx in range(cols_per_row):
            card_idx = row * cols_per_row + col_idx
            if card_idx >= num_cards:
                break  # 카드가 없으면 루프 종료
            with cols[col_idx]:
                show_quiz_card_with_flip(short_quizzes[card_idx], card_idx)

# 객관식 퀴즈를 Expander 형식으로 보여주는 함수
def show_multiple_choice_accordion(quiz, idx):
    accordion_key = f"accordion_{idx}_opened"
    answer_key = f"selected_answer_{idx}"

    # 세션 상태 키가 초기화되지 않았다면 기본값 설정
    if accordion_key not in st.session_state:
        st.session_state[accordion_key] = False
    if answer_key not in st.session_state:
        st.session_state[answer_key] = None

    # "question" 필드가 문자열인지 객체인지 확인하여 처리
    question_text = quiz.get("question", "")
    if isinstance(question_text, dict):
        question_text = question_text.get("question", "")

    # 아코디언이 열릴 때마다 선택 기록을 초기화
    with st.expander(question_text):  # 여기에 수정된 question_text를 사용
        if not st.session_state[accordion_key]:
            st.session_state[answer_key] = None
            st.session_state[accordion_key] = True

        # 라디오 버튼에서 선택할 수 있는 보기 항목
        answers = quiz.get("answers", [])  # answers 필드를 가져옴
        if isinstance(answers, list):
            options = [answer.get("answer", "보기 없음") for answer in answers]

            # 라디오 버튼 생성, 세션 상태에 저장된 선택된 값이 있으면 이를 기본값으로 설정
            user_answer = st.radio(
                "보기 중 하나를 선택하세요:",
                options,
                index=options.index(st.session_state[answer_key]) if st.session_state[answer_key] in options else 0,
                key=answer_key  # 세션 상태로 선택 기록 유지
            )

            # 선택한 답변에 대한 피드백
            explanation_shown = False
            for answer in answers:
                if user_answer == answer.get("answer"):
                    if answer.get("correct", False):  # 정답일 경우
                        st.markdown(f'<div class="correct">정답: {answer["answer"]}</div>', unsafe_allow_html=True)
                        explanation_shown = True  # 정답일 때 설명 표시
                    else:  # 오답일 경우
                        st.markdown(f'<div class="incorrect">오답: {answer["answer"]}</div>', unsafe_allow_html=True)

            # 정답일 때만 설명을 표시
            if explanation_shown:
                explanation = quiz.get("explanation", "설명이 없습니다.")
                st.markdown(f'<div class="explanation"><strong>정답 설명:</strong> {explanation}</div>', unsafe_allow_html=True)
        else:
            st.warning("보기 항목이 없습니다.")

# 객관식 퀴즈를 한 줄에 하나씩 아코디언 목록으로 보여주는 함수
def show_multiple_choice_list(quizzes):
    for idx, quiz in enumerate(quizzes):
        show_multiple_choice_accordion(quiz, idx)

# QuizRecord 페이지에서 실행되는 메인 로직
st.title("Quiz Records")

# 몽고DB에서 퀴즈 데이터를 불러옴
quizzes = load_quizzes()

if quizzes:
    # 주관식/객관식 탭 나누기
    tab1, tab2 = st.tabs(["주관식 퀴즈", "객관식 퀴즈"])

    with tab1:
        st.subheader("주관식 퀴즈")
        short_quizzes = [q for q in quizzes if q["quiz_type"] == "short"]
        if short_quizzes:
            show_short_quizzes_in_grid(short_quizzes)  # 주관식 퀴즈를 3x3 그리드로 표시
        else:
            st.warning("저장된 주관식 퀴즈가 없습니다.")

    with tab2:
        # 제목과 초기화 버튼을 같은 행에 배치
        col1, col2 = st.columns([0.8, 0.2])  # 첫 번째 칼럼은 제목, 두 번째 칼럼은 버튼
        with col1:
            st.subheader("객관식 퀴즈")
        with col2:
            if st.button("초기화"):
                # 세션 상태 초기화
                for key in st.session_state.keys():
                    del st.session_state[key]
                # 페이지 새로고침
                st.rerun()

        filtered_multiple_quizzes = [q for q in quizzes if q["quiz_type"] == "multiple"]
        show_multiple_choice_list(filtered_multiple_quizzes)  # 객관식 퀴즈만 목록으로 표시

else:
    st.warning("저장된 퀴즈가 없습니다.")
