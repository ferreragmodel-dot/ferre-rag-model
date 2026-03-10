from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware


from api.routers import llm_chat
from api.routers import llm_rag_chat, llm_agent_chat

from api.routers import test_router

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to Gianfranco Ferré research assistant"}


@app.get("/square_root/")
async def square_root(x: float = 1, y: float = 2):
    z = x**2 + y**2
    return z**0.5


# Additional routers here
app.include_router(llm_chat.router, prefix="/llm")
app.include_router(llm_rag_chat.router, prefix="/llm-rag")
app.include_router(llm_agent_chat.router, prefix="/llm-agent")
app.include_router(test_router.router, prefix="/test")