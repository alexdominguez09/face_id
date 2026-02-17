"""
FastAPI Web Application
Web interface for face recognition system.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

def create_app():
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Face Recognition System",
        description="Real-time face identification with persistent IDs",
        version="0.1.0"
    )
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Include routers
    from .routes import faces, video, api
    app.include_router(faces.router, prefix="/api", tags=["faces"])
    app.include_router(video.router, prefix="/api", tags=["video"])
    app.include_router(api.router, prefix="/api", tags=["api"])
    
    @app.get("/")
    async def root():
        """Root endpoint - serve the web interface."""
        static_dir = Path(__file__).parent / "static"
        return FileResponse(static_dir / "index.html")
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app

# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from face_recognition.config import Config
    
    uvicorn.run(
        "web.app:app",
        host=Config.WEB_HOST,
        port=Config.WEB_PORT,
        reload=True
    )