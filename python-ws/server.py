import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Constants
ZOOM_AUTH_URL = "https://zoom.us/oauth/token"
ZOOM_API_URL = "https://api.zoom.us/v2"

# Load credentials from .env
CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
REDIRECT_URI = os.getenv("ZOOM_REDIRECT_URI")

app = FastAPI()

# Store tokens in memory (for demo purposes)
tokens = {}


@app.get("/")
async def home():
    return {"message": "Zoom FastAPI Server is Running 🚀"}


@app.get("/oauth/callback")
async def zoom_oauth_callback(code: str):
    """Handles Zoom OAuth Authentication and retrieves an access token"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                ZOOM_AUTH_URL,
                params={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": REDIRECT_URI,
                },
                auth=(CLIENT_ID, CLIENT_SECRET),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="OAuth failed")

        data = response.json()
        access_token = data["access_token"]
        tokens["access_token"] = access_token  # Store the token (temporary)

        return JSONResponse(content={"message": "OAuth Success!", "access_token": access_token})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-transcript")
async def generate_transcript(request: Request):
    """Simulates transcript generation when called from the Zoom button"""
    try:
        body = await request.json()
        meeting_id = body.get("meeting_id")

        if not meeting_id:
            raise HTTPException(status_code=400, detail="Meeting ID is required")

        print(f"Transcript requested for meeting: {meeting_id}")

        # Simulate transcript processing
        return JSONResponse(content={"message": "Transcript generation started!"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
