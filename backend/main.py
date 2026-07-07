from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.configs.config import settings
from backend.api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for comparing CNN and ViT models on MedMNIST datasets with Explainable AI (XAI).",
    version="1.0.0",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medical-ai-platform-1-u7df.onrender.com/"],  # For development; can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME} API",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
