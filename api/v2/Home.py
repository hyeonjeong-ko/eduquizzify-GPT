import streamlit as st

st.set_page_config(page_title="Card Quiz", page_icon="📝", layout="centered")



# Create the quiz title
st.title("자료 유형을 선택하세요")

# Define options for the quiz
options = [
    "공식 docs 링크",
    "블로그 자료 링크",
    "개인 파일 업로드 (PDF, TXT)",
    "유튜브 영상 링크"
]

# Radio button to select the resource type
selection = st.radio("", options)

# Add a Next button
if st.button("넥스트"):
    st.write(f"선택한 자료 유형: {selection}")