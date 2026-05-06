import os
import csv
import json
import re
import time
import logging
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
from groq import Groq
from youtube_transcript_api._errors import RequestBlocked, IpBlocked
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("collect_data")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Your YouTube URLs ─────────────────────────────────────────────────────────
URLS = [
    # ── SHOULD WATCH (should_watch will be 1) ─────────────────────────
    # Your original 7 that worked
    # "https://www.youtube.com/watch?v=eyJYR_WzJYY"
    "https://www.youtube.com/watch?v=N-96Bo1EORk"
    "https://www.youtube.com/watch?v=lOKASgtr6kU"
    "https://www.youtube.com/watch?v=p5A1Pn4MGjE"
    "https://www.youtube.com/watch?v=I6qEGwpJeJc"
    "https://www.youtube.com/watch?v=WCO6v3O1OTE"
    "https://www.youtube.com/watch?v=W0sSThN2omg"
    "https://www.youtube.com/watch?v=bYSRPuDEnTg"
    "https://www.youtube.com/watch?v=DepUW80BQDw"
     "https://www.youtube.com/watch?v=bTLaCs-RWYk"
    "https://www.youtube.com/watch?v=xcJo87se5Kg"
    "https://www.youtube.com/watch?v=xBIowQ0WaR8",
    "https://www.youtube.com/watch?v=0p_881Nwoo4",
    "https://www.youtube.com/watch?v=E8cM12jRH7k",
    "https://www.youtube.com/watch?v=saSPsB-Afuw",
    "https://www.youtube.com/watch?v=lkifbWtxxlk",
    "https://www.youtube.com/watch?v=T__INNgTW1M",
    "https://www.youtube.com/watch?v=4_HOnhB64Dg",

    # New good ones — dense, technical, no fluff
    "https://www.youtube.com/watch?v=kCc8FmEb1nY",  # Andrej Karpathy: build GPT from scratch
    "https://www.youtube.com/watch?v=VMj-3S1tku0",  # Andrej Karpathy: micrograd
    # Fireship: 100 seconds of code
    "https://www.youtube.com/watch?v=Unzc731iCUY",  # MIT: how to speak
    "https://www.youtube.com/watch?v=oBt53YbR9Kk",  # dynamic programming
    "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown: neural networks
    "https://www.youtube.com/watch?v=WbzNRTTrX0g",  # system design basics
    "https://www.youtube.com/watch?v=Y8Tko2YC5hA",  # git internals

    # ── SHOULD SKIP (you'll manually set should_watch to 0) ───────────
    # Reaction videos
    "https://www.youtube.com/watch?v=PKfR6bAXr-c",  # reaction video
   # reaction video

    # Heavy sponsor / off-topic / bloated
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # rickroll — off topic
    "https://www.youtube.com/watch?v=ZSt9tm3RoUU",  # MrBeast — entertainment, no content
    "https://www.youtube.com/watch?v=BKorP55Aqvg",  # MrBeast — same

    # News channels
   # CNN news report
     # BBC news

    # Long rambling / motivational fluff
    "https://www.youtube.com/watch?v=H14bBuluwB8",  # motivational speech
    "https://www.youtube.com/watch?v=TQMbvJNRpLE",  # Tony Robbins — lots of fluff
    
    # neon man nah
    # "https://www.youtube.com/watch?v=1hN_q9Xncx0"
    # reaction vids
    # "https://www.youtube.com/watch?v=pY4j__UjTd8"
    
    #carry
    # "https://www.youtube.com/watch?v=lWBj9z2xDDs"
    # "https://www.youtube.com/watch?v=WX7DBPcsiEs"
    # "https://www.youtube.com/watch?v=5XVoRGhrhZk"
    "https://www.youtube.com/watch?v=m9s1NQG3TNY"
    # "https://www.youtube.com/watch?v=Eu173POSeF4"
   

]

# ── Output path ───────────────────────────────────────────────────────────────
# This file lives in model/ so we go one level up to reach data/
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "transcripts.csv")

CSV_FIELDS = [
    "video_id",
    "url",
    "total_minutes",
    "transcript_length",   # number of words — useful feature for model
    "difficulty",
    "should_watch",        # 1 = yes, 0 = no
    "efficiency_score",    # 0-100
    "keywords",            # comma-separated string
    "has_code",            # 1 or 0
    "has_formulas",        # 1 or 0
    "summary",
    "reason",
    "raw_transcript",      # first 2000 chars — useful for future retraining
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str | None:
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not match:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fetch_transcript(video_id: str) -> tuple[str, float]:
    """Returns (timed_text, total_minutes)"""
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id)
    lines = []
    last_end = 0.0
    for t in fetched:
        ts = seconds_to_timestamp(t.start)
        lines.append(f"[{ts}] {t.text}")
        last_end = t.start + getattr(t, "duration", 0)
    return "\n".join(lines), round(last_end / 60, 1)


def safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def label_with_groq(timed_text: str, total_minutes: float) -> dict:
    """Send transcript to Groq and get back structured labels."""
    # Trim to 6000 words to stay within token limits
    words = timed_text.split()
    trimmed = " ".join(words[:1200]) if len(words) > 1200 else timed_text

    prompt = f"""Analyze this YouTube video transcript and return ONLY valid JSON with no extra text.

{{
  "summary": "2-3 sentence summary",
  "difficulty": "Beginner or Intermediate or Advanced",
  "should_watch": true or false,
  "reason": "one sentence why or why not",
  "keywords": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "efficiency_score": <integer 0-100>,
  "has_code": true or false,
  "has_formulas": true or false
}}

Transcript ({total_minutes} min total):
{trimmed}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return safe_parse_json(response.choices[0].message.content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Check if CSV already exists so we can append without duplicating
    existing_ids = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["video_id"])
        log.info("Found existing CSV with %d entries.", len(existing_ids))

    # Open CSV in append mode
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for url in URLS:
            video_id = get_video_id(url)
            if not video_id:
                log.warning("Could not extract video ID from: %s", url)
                continue

            if video_id in existing_ids:
                log.info("Skipping (already in dataset): %s", video_id)
                continue

            log.info("Processing: %s", video_id)

            # 1. Fetch transcript
            try:
                timed_text, total_minutes = fetch_transcript(video_id)
            except NoTranscriptFound:
                log.warning("No transcript: %s", video_id)
                continue
            except VideoUnavailable:
                log.warning("Video unavailable: %s", video_id)
                continue
            except (RequestBlocked, IpBlocked) as e:
                 log.warning("YouTube temporarily blocked requests: %s", e)
                 log.warning("Sleeping for 5 minutes...")
                 time.sleep(300)
                 continue

            except Exception as e:
                log.warning("Transcript error for %s: %s", video_id, e)
                continue

            word_count = len(timed_text.split())

            # 2. Label with Groq
            try:
                labels = label_with_groq(timed_text, total_minutes)
            except Exception as e:
                log.warning("Groq labeling failed for %s: %s", video_id, e)
                continue

            # 3. Write row
            row = {
                "video_id": video_id,
                "url": url,
                "total_minutes": total_minutes,
                "transcript_length": word_count,
                "difficulty": labels.get("difficulty", "Unknown"),
                "should_watch": 1 if labels.get("should_watch", False) else 0,
                "efficiency_score": int(labels.get("efficiency_score", 0)),
                "keywords": ", ".join(labels.get("keywords", [])),
                "has_code": 1 if labels.get("has_code", False) else 0,
                "has_formulas": 1 if labels.get("has_formulas", False) else 0,
                "summary": labels.get("summary", ""),
                "reason": labels.get("reason", ""),
                "raw_transcript": timed_text[:2000],  # first 2000 chars only
            }
            writer.writerow(row)
            f.flush()  # write immediately so you can see progress

            log.info(
                "✅ Done: %s | should_watch=%s | difficulty=%s | efficiency=%s",
                video_id,
                row["should_watch"],
                row["difficulty"],
                row["efficiency_score"],
            )

            # Be polite to the API — wait 2 seconds between calls
            time.sleep(10)

    log.info("🎉 Dataset saved to: %s", OUTPUT_CSV)
    print("\n✅ Done! Open data/transcripts.csv and check the labels.")
    print("   Fix any wrong should_watch values (1=yes, 0=no) manually before training.")


if __name__ == "__main__":
    main()