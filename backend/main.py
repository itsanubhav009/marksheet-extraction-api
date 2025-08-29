# backend/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers.extract import router as extract_router

# Logging config (structured, helpful for debugging/test reports)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("marksheet-extractor")

app = FastAPI(title=settings.APP_NAME, description="Marksheet Extraction API")

# CORS - allow all origins for demo; change in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def root():
    return {"ok": True, "app": settings.APP_NAME, "version": settings.APP_VERSION}

# include router
app.include_router(extract_router, prefix="/api", tags=["extract"])

# startup/shutdown events
@app.on_event("startup")
async def on_startup():
    logger.info(f"{settings.APP_NAME} starting up...")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info(f"{settings.APP_NAME} shutting down...")
