from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
from pydantic import BaseModel
from groq import Groq
import re
import json
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    topic: str | None = None   # optional: "Does this cover X?"


class Segment(BaseModel):
    start: str
    end: str
    topic: str


class AnalyzeResponse(BaseModel):
    summary: str
    difficulty: str
    should_watch: bool
    reason: str
    useful_segments: list[Segment]
    keywords: list[str]
    efficiency_score: int
    topic_match: str | None = None   # only present when topic was given
    total_duration_minutes: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str | None:
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
        # handle youtu.be short links too
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds → MM:SS string."""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_timed_text(fetched) -> tuple[str, float]:
    """
    Build a timestamped transcript string so the LLM
    can cite REAL timestamps instead of hallucinating them.
    Returns (timed_text, total_duration_seconds).
    """
    lines = []
    last_end = 0.0
    for t in fetched:
        ts = seconds_to_timestamp(t.start)
        lines.append(f"[{ts}] {t.text}")
        last_end = t.start + getattr(t, "duration", 0)
    return "\n".join(lines), last_end


def chunk_text(text: str, chunk_size: int = 8000) -> list[str]:
    """Split text into chunks of ~chunk_size characters on word boundaries."""
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def condense_long_transcript(timed_text: str) -> str:
    """
    For very long transcripts, do a cheap summarise-per-chunk pass first
    so the final prompt stays well within context limits.
    """
    chunks = chunk_text(timed_text, chunk_size=8000)
    summaries = []
    for chunk in chunks:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this transcript chunk in 4-5 sentences. "
                    "Preserve exact timestamps, key topics, any code snippets "
                    "or formulas mentioned, and difficulty signals.\n\n"
                    + chunk
                ),
            }],
        )
        summaries.append(r.choices[0].message.content.strip())
    return "\n\n".join(summaries)


def safe_parse_json(raw: str) -> dict:
    """Strip markdown fences then parse JSON."""
    raw = raw.strip()
    # remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    # 1. Extract video ID
    video_id = get_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Could not extract video ID from URL.")

    # 2. Fetch transcript
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript available for this video.")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable or private.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript fetch failed: {e}")

    # 3. Build timestamped text
    timed_text, total_seconds = build_timed_text(fetched)
    total_minutes = round(total_seconds / 60, 1)

    # 4. Condense if very long (>80k chars ≈ long lecture)
    prompt_text = timed_text if len(timed_text) <= 80_000 else condense_long_transcript(timed_text)

    # 5. Build topic-match instruction (optional)
    topic_instruction = ""
    if req.topic:
        topic_instruction = (
            f'\n  "topic_match": "one sentence: does this video cover \\"{req.topic}\\"? yes/no + why",'
        )

    # 6. Call Groq
    prompt = f"""Analyze this video transcript and return ONLY valid JSON with no extra text.
Use the [MM:SS] timestamps in the transcript to fill useful_segments accurately.

{{
  "summary": "2-3 sentence summary",
  "difficulty": "Beginner | Intermediate | Advanced",
  "should_watch": true or false,
  "reason": "one sentence why or why not",
  "useful_segments": [{{"start": "MM:SS", "end": "MM:SS", "topic": "what is covered"}}],
  "keywords": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "efficiency_score": <integer 0-100 — ratio of genuinely useful minutes to total>,{topic_instruction}
  "has_code": true or false,
  "has_formulas": true or false
}}

Transcript ({total_minutes} min total):
{prompt_text}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # 7. Parse and return
    raw = response.choices[0].message.content
    try:
        data = safe_parse_json(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    return AnalyzeResponse(
        summary=data.get("summary", ""),
        difficulty=data.get("difficulty", "Unknown"),
        should_watch=bool(data.get("should_watch", False)),
        reason=data.get("reason", ""),
        useful_segments=[
            Segment(start=s["start"], end=s["end"], topic=s["topic"])
            for s in data.get("useful_segments", [])
        ],
        keywords=data.get("keywords", []),
        efficiency_score=int(data.get("efficiency_score", 0)),
        topic_match=data.get("topic_match"),
        total_duration_minutes=total_minutes,
    )