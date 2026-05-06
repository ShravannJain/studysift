const API = "http://localhost:8000";
let currentUrl = "";
let currentDecision = "";

// ── State helpers ─────────────────────────────────────────────────────────────

const states = ["loading", "nonyoutube", "error", "result"];

function showState(name) {
  states.forEach(s => {
    document.getElementById(`state-${s}`).style.display = s === name ? "block" : "none";
  });

  const dot = document.getElementById("status-dot");
  dot.classList.toggle("active", name === "result");
}

function setError(msg) {
  document.getElementById("error-msg").textContent = msg;
  showState("error");
}

// ── Analyze ───────────────────────────────────────────────────────────────────

async function analyze(topic = null) {
  showState("loading");

  let tab;
  try {
    [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  } catch {
    setError("Could not access the current tab.");
    return;
  }

  currentUrl = tab.url || "";

  // Non-YouTube guard
  if (!currentUrl.includes("youtube.com/watch") && !currentUrl.includes("youtu.be/")) {
    showState("nonyoutube");
    return;
  }

  try {
    const res = await fetch(`${API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentUrl, topic }),
    });

    if (!res.ok) {
      let detail = "Server returned an error.";
      try { detail = (await res.json()).detail || detail; } catch {}
      setError(detail);
      return;
    }

    const data = await res.json();
    renderResult(data);

  } catch (e) {
    const msg = e.message?.includes("Failed to fetch")
      ? "Cannot reach backend. Is uvicorn running on :8000?"
      : (e.message || "Unknown error");
    setError(msg);
  }
}

// ── Render result ─────────────────────────────────────────────────────────────

function renderResult(data) {
  currentDecision = data.should_watch ? "watch" : "skip";

  // Verdict card
  const card  = document.getElementById("verdict-card");
  const label = document.getElementById("verdict-label");
  const badge = document.getElementById("verdict-badge");
  card.className  = `verdict-card anim ${currentDecision}`;
  label.className = `verdict-label ${currentDecision}`;
  badge.className = `verdict-badge ${currentDecision}`;
  label.textContent = data.should_watch ? "WATCH" : "SKIP";
  badge.textContent = data.should_watch ? "✓ WORTH IT" : "✕ SKIP IT";
  document.getElementById("verdict-reason").textContent = data.reason || "";

  // Meta chips
  document.getElementById("meta-efficiency").textContent =
    `${data.efficiency_score ?? 0}% efficient`;
  document.getElementById("meta-duration").textContent =
    `${data.total_duration_minutes ?? "?"} min`;
  const diffEl = document.getElementById("meta-difficulty");
  diffEl.textContent = data.difficulty || "";

  // Topic match
  const tmCard = document.getElementById("topic-match-card");
  const tmText = document.getElementById("topic-match-text");
  if (data.topic_match) {
    tmText.textContent = data.topic_match;
    tmCard.style.display = "block";
  } else {
    tmCard.style.display = "none";
  }

  // Keywords
  const kwWrap = document.getElementById("keywords-wrap");
  kwWrap.innerHTML = (data.keywords || [])
    .map(k => `<span class="kw-pill">${escHtml(k)}</span>`)
    .join("");

  // Segments
  const segList = document.getElementById("segments-list");
  const segSection = document.getElementById("segments-section");
  if (data.useful_segments?.length) {
    segList.innerHTML = data.useful_segments
      .map(s => `
        <div class="segment-card">
          <div class="seg-time">${escHtml(s.start)}–${escHtml(s.end)}</div>
          <div class="seg-topic">${escHtml(s.topic)}</div>
        </div>`)
      .join("");
    segSection.style.display = "block";
  } else {
    segSection.style.display = "none";
  }

  // Reset feedback
  document.getElementById("feedback-msg").textContent = "";
  document.getElementById("thumb-up").className   = "thumb-btn";
  document.getElementById("thumb-down").className = "thumb-btn";

  // Reset topic input
  document.getElementById("topic-input").value = "";
  document.getElementById("topic-btn").disabled = false;

  showState("result");
}

// ── Feedback ──────────────────────────────────────────────────────────────────

async function sendFeedback(wasCorrect) {
  const upBtn   = document.getElementById("thumb-up");
  const downBtn = document.getElementById("thumb-down");
  const msg     = document.getElementById("feedback-msg");

  upBtn.className   = wasCorrect ? "thumb-btn active-up"   : "thumb-btn";
  downBtn.className = wasCorrect ? "thumb-btn"              : "thumb-btn active-down";

  try {
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: currentUrl,
        decision: currentDecision,
        was_correct: wasCorrect,
      }),
    });
    msg.textContent = wasCorrect ? "Got it!" : "Noted — will learn.";
  } catch {
    msg.textContent = "Feedback failed.";
  }
}

// ── Util ──────────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Events ────────────────────────────────────────────────────────────────────

document.getElementById("thumb-up").addEventListener("click",   () => sendFeedback(true));
document.getElementById("thumb-down").addEventListener("click", () => sendFeedback(false));

document.getElementById("topic-btn").addEventListener("click", () => {
  const topic = document.getElementById("topic-input").value.trim();
  if (!topic) return;
  document.getElementById("topic-btn").disabled = true;
  analyze(topic);
});

document.getElementById("topic-input").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("topic-btn").click();
});

document.getElementById("retry-btn").addEventListener("click", () => analyze());

// ── Boot ──────────────────────────────────────────────────────────────────────
analyze();