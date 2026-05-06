from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
from pydantic import BaseModel
from groq import Groq
import re
import json
import os
import hashlib
import time
import logging
import csv
from datetime import datetime
from functools import lru_cache
from typing import Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("studysift")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Cache setup ───────────────────────────────────────────────────────────────
# Tries Redis first; falls back to in-memory dict automatically.
# To use Redis: pip install redis  and  set REDIS_URL=redis://localhost:6379 in .env

CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", 60 * 60 * 24 * 7))  # 7 days default

redis_client = None
try:
    import redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
    redis_client.ping()
    log.info("✅ Redis cache connected: %s", redis_url)
except Exception as e:
    log.warning("⚠️  Redis unavailable (%s) — using in-memory cache instead.", e)
    redis_client = None

# In-memory fallback: {cache_key: {"data": {...}, "expires_at": float}}
_memory_cache: dict = {}


def _make_cache_key(video_id: str, topic: Optional[str]) -> str:
    raw = f"{video_id}:{(topic or '').strip().lower()}"
    return "studysift:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_get(key: str) -> Optional[dict]:
    # 1. Try Redis
    if redis_client:
        try:
            val = redis_client.get(key)
            if val:
                log.info("cache HIT (redis) %s", key)
                return json.loads(val)
        except Exception as e:
            log.warning("Redis GET failed: %s", e)

    # 2. In-memory fallback
    entry = _memory_cache.get(key)
    if entry and entry["expires_at"] > time.time():
        log.info("cache HIT (memory) %s", key)
        return entry["data"]
    if entry:
        del _memory_cache[key]  # expired

    return None


def cache_set(key: str, data: dict, ttl: int = CACHE_TTL) -> None:
    payload = json.dumps(data)

    # 1. Try Redis
    if redis_client:
        try:
            redis_client.setex(key, ttl, payload)
            log.info("cache SET (redis) %s  TTL=%ds", key, ttl)
            return
        except Exception as e:
            log.warning("Redis SET failed: %s", e)

    # 2. In-memory fallback (cap at 500 entries to avoid unbounded growth)
    if len(_memory_cache) >= 500:
        oldest = min(_memory_cache, key=lambda k: _memory_cache[k]["expires_at"])
        del _memory_cache[oldest]

    _memory_cache[key] = {"data": data, "expires_at": time.time() + ttl}
    log.info("cache SET (memory) %s  TTL=%ds", key, ttl)


def cache_delete(key: str) -> None:
    if redis_client:
        try:
            redis_client.delete(key)
        except Exception:
            pass
    _memory_cache.pop(key, None)


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    topic: str | None = None
    force_refresh: bool = False   # set true to bypass cache


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
    topic_match: str | None = None
    total_duration_minutes: float
    cached: bool = False   # tells the frontend whether this came from cache

class FeedbackRequest(BaseModel):
    url: str
    decision: str        # "watch" or "skip"
    was_correct: bool   
# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str | None:
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_timed_text(fetched) -> tuple[str, float]:
    lines = []
    last_end = 0.0
    for t in fetched:
        ts = seconds_to_timestamp(t.start)
        lines.append(f"[{ts}] {t.text}")
        last_end = t.start + getattr(t, "duration", 0)
    return "\n".join(lines), last_end


def chunk_text(text: str, chunk_size: int = 8000) -> list[str]:
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
    chunks = chunk_text(timed_text, chunk_size=8000)
    summaries = []
    for chunk in chunks:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=120,
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
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    # 1. Extract video ID
    video_id = get_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Could not extract video ID from URL.")

    cache_key = _make_cache_key(video_id, req.topic)

    # 2. Check cache (skip if force_refresh=true)
    if not req.force_refresh:
        cached = cache_get(cache_key)
        if cached:
            return AnalyzeResponse(**cached, cached=True)

    # 3. Fetch transcript
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript available for this video.")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable or private.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcript fetch failed: {e}")

    # 4. Build timestamped text
    timed_text, total_seconds = build_timed_text(fetched)
    total_minutes = round(total_seconds / 60, 1)

    # 5. Condense if very long
    prompt_text = " ".join(timed_text.split()[:1500])

    # 6. Build prompt
    topic_instruction = ""
    if req.topic:
        topic_instruction = (
            f'\n  "topic_match": "one sentence: does this video cover \\"{req.topic}\\"? yes/no + why",'
        )

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

    # 7. Call Groq
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # 8. Parse response
    raw = response.choices[0].message.content
    try:
        data = safe_parse_json(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    # 9. Build result dict and store in cache
    result = {
        "summary": data.get("summary", ""),
        "difficulty": data.get("difficulty", "Unknown"),
        "should_watch": bool(data.get("should_watch", False)),
        "reason": data.get("reason", ""),
        "useful_segments": [
            {"start": s["start"], "end": s["end"], "topic": s["topic"]}
            for s in data.get("useful_segments", [])
        ],
        "keywords": data.get("keywords", []),
        "efficiency_score": int(data.get("efficiency_score", 0)),
        "topic_match": data.get("topic_match"),
        "total_duration_minutes": total_minutes,
    }
    cache_set(cache_key, result)

    return AnalyzeResponse(**result, cached=False)


@app.delete("/cache/{video_id}")
async def invalidate_cache(video_id: str, topic: str | None = None):
    """Manually bust cache for a video (useful during development)."""
    key = _make_cache_key(video_id, topic)
    cache_delete(key)
    return {"deleted": key}


@app.get("/cache/stats")
async def cache_stats():
    """Quick health check — shows memory cache size and Redis status."""
    redis_ok = False
    if redis_client:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            pass
    return {
        "redis_connected": redis_ok,
        "memory_cache_entries": len(_memory_cache),
        "cache_ttl_seconds": CACHE_TTL,
    }
    
@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    row = {
        "timestamp": datetime.now().isoformat(),
        "url": req.url,
        "decision": req.decision,
        "was_correct": req.was_correct,
    }
    path = "data/feedback.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    return {"ok": True}