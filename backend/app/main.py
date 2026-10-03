from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import task1


app = FastAPI(
    title="Generative AI Assignment API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task1.router)


@app.get("/")
def root():
    return {
        "name": "Generative AI Assignment API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
