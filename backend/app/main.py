from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import availability
from .routes import router

app = FastAPI(title="GenAI Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/health")
def health():
    return {"status": "ok", "models": availability()}