from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from explanation_module import explain_topic
from learning_path import get_learning_recommendations
from qna import answer_question
from quiz_module import generate_quiz
from summary_module import summarize_text


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="EduGenie",
    version="1.0.0",
    description="Google Gemini powered AI learning assistant",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=30000,
        description="User input text",
    )


class LearningRequest(TextRequest):
    level: str = Field(
        default="beginner",
        min_length=1,
        max_length=40,
    )


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def run_safely(function, *args, **kwargs):
    """
    Execute a service function and convert unexpected
    application errors into an HTTP 502 response.
    """
    try:
        return function(*args, **kwargs)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "EduGenie",
    }


# ---------------------------------------------------------
# Question Answering
# ---------------------------------------------------------

@app.post("/qa")
async def qa(payload: TextRequest):
    answer = run_safely(
        answer_question,
        payload.text,
    )

    return {
        "result": answer,
    }


# ---------------------------------------------------------
# Concept Explanation
# ---------------------------------------------------------

@app.post("/explain")
async def explain(payload: TextRequest):
    explanation = run_safely(
        explain_topic,
        payload.text,
    )

    return {
        "result": explanation,
    }


# ---------------------------------------------------------
# Quiz Generation
# ---------------------------------------------------------

@app.post("/quiz")
async def quiz(payload: TextRequest):
    quiz = run_safely(
        generate_quiz,
        payload.text,
    )

    return quiz.model_dump()


# ---------------------------------------------------------
# Summarization
# ---------------------------------------------------------

@app.post("/summarize")
async def summarize(payload: TextRequest):
    summary = run_safely(
        summarize_text,
        payload.text,
    )

    return {
        "result": summary,
    }


# ---------------------------------------------------------
# Learning Recommendations
# ---------------------------------------------------------

@app.post("/learn/recommendations")
async def recommendations(payload: LearningRequest):
    recommendations = run_safely(
        get_learning_recommendations,
        payload.text,
        payload.level,
    )

    return {
        "result": recommendations,
    }