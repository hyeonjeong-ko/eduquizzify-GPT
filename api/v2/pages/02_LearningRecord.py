import streamlit as st
from pymongo import MongoClient

# MongoDB 클라이언트 생성 및 데이터베이스 연결 함수
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")  # MongoDB 서버의 URL
    return client

# 몽고DB에서 학습 카드를 불러오는 함수
def load_learning_cards():
    client = get_mongo_client()
    db = client["quiz_db"]
    collection = db["learning_cards"]
    learning_cards = list(collection.find())  # 학습 카드 목록을 가져옴
    client.close()  # 연결 닫기
    return learning_cards

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

.explanation {
  background-color: #f1f3f5;
  color: black;  /* 설명 부분의 글씨를 검정으로 설정 */
  padding: 10px;
  margin-top: 10px;
  border-left: 5px solid #2980b9;
  border-radius: 5px;
}
</style>
"""

# 학습 카드를 플립 카드로 보여주는 함수
def show_learning_card_with_flip(learning_card, idx):
    st.markdown(card_flip_css, unsafe_allow_html=True)

    statement_text = learning_card.get("statement", "No statement available.")
    question_text = learning_card.get("question", "No question available.")
    explanation_text = learning_card.get("explanation", "No explanation available.")

    flip_card_html = f"""
    <div class="flip-card">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <p>{statement_text}</p>
        </div>
        <div class="flip-card-back">
          <h5>질문</h5>
          <p>{question_text}</p>
          <div class="explanation">
            <strong>설명:</strong> {explanation_text}
          </div>
        </div>
      </div>
    </div>
    """
    st.markdown(flip_card_html, unsafe_allow_html=True)

# 학습 카드를 3x3 그리드로 나열하는 함수
def show_learning_cards_in_grid(learning_cards):
    num_cards = len(learning_cards)
    cols_per_row = 3  # 한 줄에 3개의 카드 배치
    rows = num_cards // cols_per_row + (1 if num_cards % cols_per_row else 0)  # 필요한 행 수 계산

    for row in range(rows):
        cols = st.columns(3)  # 3개의 열 생성
        for col_idx in range(cols_per_row):
            card_idx = row * cols_per_row + col_idx
            if card_idx >= num_cards:
                break  # 카드가 없으면 루프 종료
            with cols[col_idx]:
                show_learning_card_with_flip(learning_cards[card_idx], card_idx)

# 학습 기록 페이지에서 실행되는 메인 로직
st.title("Learning Records")

# 몽고DB에서 학습 카드 데이터를 불러옴
learning_cards = load_learning_cards()

if learning_cards:
    # 학습 카드 표시
    st.subheader("학습 카드")
    show_learning_cards_in_grid(learning_cards)  # 학습 카드 3x3 그리드로 표시
else:
    st.warning("저장된 학습 카드가 없습니다.")
