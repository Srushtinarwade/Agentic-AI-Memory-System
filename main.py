from fastapi import FastAPI

from app.router.memory import router as memory_router
from app.router.chat import router as chat_router

app = FastAPI(title="Agentic AI Memory System")

app.include_router(memory_router)
app.include_router(chat_router)


@app.get("/")
def read_root():
    return {"message": "Agentic AI Memory System is running."}