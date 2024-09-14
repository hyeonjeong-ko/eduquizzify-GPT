from fastapi import FastAPI
from summarize import router as summarize_router
from quiz import router as quiz_router

app = FastAPI()

# 두 개의 라우터를 메인 앱에 등록
app.include_router(summarize_router, prefix="/files")  # 여기서 /files로 요약 관련 기능을 처리
app.include_router(quiz_router, prefix="/files")  # 여기서 /files로 요약 관련 기능을 처리

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
