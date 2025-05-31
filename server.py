from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

# Load environment variables
load_dotenv()

# OpenAI config
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key, base_url=api_base)

# FastAPI app
app = FastAPI()

# Allow CORS (for local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/meta_critique")
async def meta_critique(request: Request):
    data = await request.json()
    feedback = data.get("feedback", "")

    system_prompt = """
You are a helpful assistant that critiques student feedback on UX design presentations.
Help students make their feedback clearer, more specific, and more constructive.

Focus on areas like: goals, structure, navigation, visual design, text clarity, accessibility, interactions, responsiveness, and implementation readiness.

Respond in this format:

Overall Assessment:
...

Suggestions for Improvement:
- ...
- ...
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": feedback}
            ]
        )
        critique = response.choices[0].message.content
        return JSONResponse(content={"meta_feedback": critique})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})