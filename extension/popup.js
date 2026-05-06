const API = "http://localhost:8000";
let currentUrl = "";
let currentDecision = "";

async function analyze(topic = null) {
  document.getElementById("status").textContent = "Analyzing...";
  document.getElementById("result").style.display = "none";
  document.getElementById("error").textContent = "";

  try {
    const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
    currentUrl = tab.url;

    const res = await fetch(`${API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentUrl, topic }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    renderResult(data);

  } catch (e) {
    document.getElementById("status").textContent = "";
    document.getElementById("error").textContent =
      e.message.includes("Failed to fetch")
        ? "Cannot reach backend. Is uvicorn running?"
        : e.message;
  }
}

function renderResult(data) {
  document.getElementById("status").textContent = "";
  document.getElementById("result").style.display = "block";

  currentDecision = data.should_watch ? "watch" : "skip";

  const dec = document.getElementById("decision");
  dec.textContent = data.should_watch ? "WATCH" : "SKIP";
  dec.className = data.should_watch ? "watch" : "skip";

  document.getElementById("reason").textContent = data.reason;
  document.getElementById("score").textContent =
    `Efficiency: ${data.efficiency_score}%  •  ${data.total_duration_minutes} min video` +
    (data.cached ? "  •  cached" : "");

  // keywords
  const kwDiv = document.getElementById("keywords");
  kwDiv.innerHTML = data.keywords
    .map(k => `<span class="kw">${k}</span>`)
    .join("");

  // segments
  document.getElementById("segments").innerHTML =
    data.useful_segments
      .map(s => `<li><b>${s.start}–${s.end}</b> ${s.topic}</li>`)
      .join("");

  // topic match
  if (data.topic_match) {
    const tm = document.getElementById("topic-match");
    tm.textContent = data.topic_match;
    tm.style.display = "block";
  }
}

async function sendFeedback(was_correct) {
  try {
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: currentUrl,
        decision: currentDecision,
        was_correct,
      }),
    });
    document.getElementById("feedback-msg").textContent =
      was_correct ? "Got it!" : "Noted, will learn.";
  } catch {
    document.getElementById("feedback-msg").textContent = "Feedback failed.";
  }
}

document.getElementById("thumb-up").addEventListener("click", () => sendFeedback(true));
document.getElementById("thumb-down").addEventListener("click", () => sendFeedback(false));
document.getElementById("topic-btn").addEventListener("click", () => {
  const topic = document.getElementById("topic-input").value.trim();
  if (topic) analyze(topic);
});

analyze();