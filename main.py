from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
import re
import json
import os
load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

@app.post("/analyze")
async def analyze(data: dict):
    url = data["url"]
    video_id = get_video_id(url)
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id)
    text = " ".join([t.text for t in fetched])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        temperature=0.3,
        messages=[{"role": "user", "content": f"""Analyze this video transcript and return ONLY valid JSON with no extra text:
{{
  "summary": "2-3 sentence summary",
  "difficulty": "Beginner | Intermediate | Advanced",
  "should_watch": true or false,
  "reason": "one sentence why or why not",
  "useful_segments": [{{"start": "MM:SS", "end": "MM:SS", "topic": "what's covered"}}],
  "keywords": ["topic1", "topic2", "topic3"],
  "efficiency_score": 0
}}
Transcript: {text[:4000]}"""}]
    )

    raw = response.choices[0].message.content
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)