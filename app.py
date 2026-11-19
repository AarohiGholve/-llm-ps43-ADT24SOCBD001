import difflib
import html
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
# Speed knob for Gemini 3.x models: "low", "medium" (default) or "high".
# "low" is fastest and plenty for a short writing task. (3.8 Flash has no "minimal".)
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
# Optional backup model used if the main one stays overloaded (e.g. gemini-2.5-flash).
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()

RETRY_CODES = {429, 500, 503, 504}  # temporary errors worth retrying
MAX_ATTEMPTS = 3

CREATOR = "Aarohi Gholve"

# Rough size of a typical answer, used to estimate progress. It self-corrects
# after every run (see _expected_chars in session state).
DEFAULT_EXPECTED_CHARS = 1500


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("profile_optimizer")
    if not logger.handlers:  # Streamlit re-runs this file; add the handler once.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = _get_logger()


@st.cache_resource(show_spinner=False)
def log_startup() -> bool:
    log.info(
        "App ready | model=%s | thinking_level=%s | api_key=%s",
        GEMINI_MODEL or "(not set)",
        GEMINI_THINKING_LEVEL,
        "set" if GEMINI_API_KEY else "MISSING",
    )
    return True


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def build_prompt(
    current_headline: str,
    current_about: str,
    skills: str,
    target_role: str,
) -> str:
    return f"""You are a practical LinkedIn profile writing assistant for students.

Use the information below to create:
1. Exactly 3 distinct LinkedIn headline options.
2. One polished About section tailored to the target role.
3. An improved skills list for the target role.

Keep the writing specific, professional, and easy to scan. Use relevant
keywords naturally. Do not invent experience, qualifications, achievements,
or metrics that are not provided. If the student has limited experience,
emphasize skills, interests, and goals honestly.

Skills rules:
- KEEP: skills from the student's list that are worth keeping, copied exactly.
- ADD: up to 6 skills that employers expect for the target role AND that the
  student's headline, About section or skills already support (for example a
  more specific name for something they mention). Never add tools or
  experience the student did not mention. Write "none" if nothing is supported.
- REMOVE: skills from the student's list that are vague, duplicated or
  irrelevant to the target role, copied exactly. Write "none" if nothing.
- Each skill is a short plain name (1-4 words), no explanations.

Return only the requested content in exactly this format:
## Headline options
1. ...
2. ...
3. ...

## About section
...

## Skills
KEEP: skill one, skill two
ADD: skill three, skill four
REMOVE: skill five

Current headline:
{current_headline}

Current About section:
{current_about}

Skills:
{skills}

Target role:
{target_role}
"""


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    # One client for the whole session instead of a new connection per click.
    return genai.Client(api_key=api_key)


def build_config(use_thinking_level: bool = True):
    # thinking_level only exists on Gemini 3.x; never send it to a fallback model.
    if use_thinking_level and GEMINI_THINKING_LEVEL in {"low", "medium", "high"}:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL)
        )
    return None


def stream_profile(
    api_key: str,
    model: str,
    current_headline: str,
    current_about: str,
    skills: str,
    target_role: str,
):
    """Yield text chunks as Gemini produces them.

    Temporary errors (503 overloaded, 429 rate limit, 500/504) are retried with a
    short wait, then the optional fallback model is tried. Everything is logged.
    """
    prompt = build_prompt(
        current_headline=current_headline,
        current_about=current_about,
        skills=skills,
        target_role=target_role,
    )
    client = get_client(api_key)
    started = time.perf_counter()
    first_token_at = None
    chunks = 0
    total_chars = 0

    attempts = [(model, True)] * MAX_ATTEMPTS
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != model:
        attempts.append((GEMINI_FALLBACK_MODEL, False))

    last_exc = None
    for number, (current_model, use_thinking) in enumerate(attempts, start=1):
        log.info(
            "Request sent (attempt %d/%d) | model=%s | thinking_level=%s | prompt=%d chars",
            number,
            len(attempts),
            current_model,
            GEMINI_THINKING_LEVEL if use_thinking else "n/a",
            len(prompt),
        )
        try:
            for chunk in client.models.generate_content_stream(
                model=current_model,
                contents=prompt,
                config=build_config(use_thinking),
            ):
                if chunk.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - started
                        log.info("First token after %.2fs", first_token_at)
                    chunks += 1
                    total_chars += len(chunk.text)
                    yield chunk.text
            break  # finished without an API error
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            log.warning(
                "Attempt %d failed after %.2fs | %s | code=%s | %s",
                number,
                time.perf_counter() - started,
                type(exc).__name__,
                code,
                exc,
            )
            # Cannot cleanly retry once text has started, or for non-temporary errors.
            if total_chars or code not in RETRY_CODES:
                raise
            last_exc = exc
            if number < len(attempts):
                wait = 2 * number
                log.info("Temporary error, retrying in %ds...", wait)
                time.sleep(wait)
    else:
        log.error("All %d attempts failed", len(attempts))
        raise last_exc

    total = time.perf_counter() - started
    if not total_chars:
        log.warning("Gemini returned no text after %.2fs", total)
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    log.info(
        "Done | %d chunks | %d chars | total %.2fs (waiting %.2fs + writing %.2fs)",
        chunks,
        total_chars,
        total,
        first_token_at or 0.0,
        total - (first_token_at or 0.0),
    )


# --------------------------------------------------------------------------
# Helpers for presenting the result
# --------------------------------------------------------------------------
def _clean(text: str) -> str:
    return re.sub(r"\*\*|__", "", text).strip()


def _section_map(text: str) -> dict:
    sections, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^\s*#{1,6}\s*(.+?)\s*$", line)
        if m:
            current = m.group(1).lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def split_skills(raw: str) -> list:
    """'a, b; c' -> ['a', 'b', 'c'] (de-duplicated, order kept, 'none' ignored)."""
    seen, items = set(), []
    for part in re.split(r"[,\n;|•]+", raw or ""):
        part = _clean(part).strip(" -*·.")
        if part and part.lower() not in {"none", "n/a"} and part.lower() not in seen:
            seen.add(part.lower())
            items.append(part)
    return items


def parse_result(text: str) -> dict:
    """Split the model output into headlines, About text and skill lists."""
    sections = _section_map(text)
    pick = lambda key: next((v for k, v in sections.items() if key in k), "")
    headlines = [
        _clean(m.group(1))
        for m in re.finditer(r"(?m)^\s*\d+[.)]\s+(.+)$", pick("headline"))
    ][:3]
    skills = None
    skills_raw = pick("skill")
    if skills_raw:
        found = {}
        for line in skills_raw.splitlines():
            m = re.match(r"^[\s\-*•]*(keep|add|remove)[\s*]*:[\s*]*(.*)$", line, re.I)
            if m:
                found[m.group(1).lower()] = split_skills(m.group(2))
        if found:
            skills = {k: found.get(k, []) for k in ("keep", "add", "remove")}
    return {"headlines": headlines, "about": pick("about").strip(), "skills": skills}


def skill_diff(original_text: str, suggested: dict) -> dict:
    """Compare the student's skills with the suggestions, in code (not by trusting the model)."""
    original = split_skills(original_text)
    original_l = {s.lower() for s in original}
    removed_l = {s.lower() for s in suggested["remove"]}
    added_l = set()
    kept, removed, added = [], [], []
    for skill in original:
        # Only removed if the model said so; anything it forgot to mention is kept.
        (removed if skill.lower() in removed_l else kept).append(skill)
    for skill in suggested["add"] + suggested["keep"]:
        low = skill.lower()
        if low not in original_l and low not in added_l:
            added_l.add(low)
            added.append(skill)
    return {"original": original, "kept": kept, "added": added, "removed": removed}


def chips(items: list, kind: str) -> str:
    return "".join(
        f'<span class="chip {kind}"><span class="t">{html.escape(i)}</span></span>'
        for i in items
    )


def word_diff_html(before: str, after: str) -> str:
    a, b = before.split(), after.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(html.escape(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return f"<p class='diff'>{' '.join(out)}</p>"


_COPY_TEMPLATE = """
<style>
  body {margin: 0; font-family: Inter, -apple-system, 'Segoe UI', Roboto, sans-serif;}
  button {
    width: 100%; height: 40px; border-radius: 999px; cursor: pointer;
    border: 1.5px solid #0A66C2; background: #fff; color: #0A66C2;
    font-weight: 600; font-size: 13px; transition: all .15s ease;
  }
  button:hover {background: #E8F1FB;}
  button.ok {background: #1FA855; border-color: #1FA855; color: #fff;}
</style>
<button id="b" type="button"></button>
<script>
  const text = __PAYLOAD__, label = __LABEL__, b = document.getElementById('b');
  b.textContent = label;
  b.onclick = async () => {
    try { await navigator.clipboard.writeText(text); }
    catch (e) {
      const t = document.createElement('textarea');
      t.value = text; document.body.appendChild(t); t.select();
      document.execCommand('copy'); t.remove();
    }
    b.textContent = 'Copied \\u2713'; b.className = 'ok';
    setTimeout(() => { b.textContent = label; b.className = ''; }, 1600);
  };
</script>
"""


def copy_button(text: str, label: str = "Copy", height: int = 46) -> None:
    payload = json.dumps(text).replace("</", "<\\/")
    page = _COPY_TEMPLATE.replace("__PAYLOAD__", payload).replace("__LABEL__", json.dumps(label))
    components.html(page, height=height)


def to_html_paragraphs(text: str) -> str:
    safe = html.escape(text.strip())
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", safe) if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)


def text_block(text: str) -> str:
    return to_html_paragraphs(text) or "<p class='muted'>Nothing added.</p>"


# Share of the outline given to each edge: top, right, bottom, left (clockwise).
EDGE_WEIGHTS = (0.36, 0.14, 0.36, 0.14)


def edge_fills(progress: float):
    """Turn overall progress (0-1) into a fill % for each of the four edges."""
    fills, remaining = [], max(0.0, min(progress, 1.0))
    for weight in EDGE_WEIGHTS:
        take = min(remaining, weight)
        fills.append(take / weight * 100)
        remaining -= take
    return fills


def render_live_text(text: str, done: bool) -> str:
    if not text:
        return "<p class='muted'>Gemini is thinking&hellip;</p>"
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts.append(f"<div class='live-h'>{html.escape(line.lstrip('#').strip())}</div>")
        else:
            safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(line))
            parts.append(f"<p>{safe}</p>")
    if not done:
        parts.append("<span class='cursor'></span>")
    return "".join(parts)


def live_box(text: str, progress: float, elapsed: float, done: bool = False) -> str:
    top, right, bottom, left = edge_fills(progress)
    status = "Done" if done else ("Writing" if text else "Thinking")
    css = "ring" + (" done" if done else "") + ("" if text else " thinking")
    # Single line on purpose: blank lines would end the HTML block in Markdown.
    return (
        f'<div class="{css}">'
        f'<i class="edge top" style="width:{top:.1f}%"></i>'
        f'<i class="edge right" style="height:{right:.1f}%"></i>'
        f'<i class="edge bottom" style="width:{bottom:.1f}%"></i>'
        f'<i class="edge left" style="height:{left:.1f}%"></i>'
        f'<div class="ring-head"><span class="ring-status">{status}</span>'
        f'<span class="ring-meta">{int(progress * 100)}% &middot; {elapsed:.1f}s</span></div>'
        f'<div class="ring-body"><div>{render_live_text(text, done)}</div></div>'
        f"</div>"
    )


# --------------------------------------------------------------------------
# Page + theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="LinkedIn Profile Optimizer",
    page_icon="💼",
    layout="wide",
)
log_startup()

# Tri-color palette:  Blue #0A66C2 (action)  |  Navy #0B2545 (ink)  |  Amber #F2A93B (accent)
# Neutrals: #F4F6F9 page, #FFFFFF cards, #DCE4EE borders, #5B6B7F muted text.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --blue: #0A66C2;
        --blue-dark: #084C91;
        --blue-soft: #E8F1FB;
        --navy: #0B2545;
        --amber: #F2A93B;
        --amber-soft: #FEF4E1;
        --bg: #F4F6F9;
        --card: #FFFFFF;
        --line: #DCE4EE;
        --muted: #5B6B7F;
        --green: #1FA855;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: var(--navy);
    }
    .stApp {background: var(--bg);}
    header[data-testid="stHeader"] {background: transparent;}
    #MainMenu, footer {visibility: hidden;}
    .block-container {max-width: 1080px; padding-top: 1.6rem; padding-bottom: 1.5rem;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, var(--navy) 0%, #12386B 55%, var(--blue) 100%);
        border-radius: 20px; padding: 2.2rem 2.4rem 2rem;
        box-shadow: 0 12px 30px rgba(11, 37, 69, 0.18);
        border-bottom: 5px solid var(--amber);
        margin-bottom: 1.2rem;
    }
    .hero::after {
        content: ""; position: absolute; right: -60px; top: -60px;
        width: 240px; height: 240px; border-radius: 50%;
        background: radial-gradient(circle, rgba(242,169,59,0.35), transparent 70%);
    }
    .hero .pill {
        display: inline-block; background: var(--amber); color: var(--navy);
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.28rem 0.7rem; border-radius: 999px;
    }
    .hero h1 {
        color: #fff !important; font-weight: 800; letter-spacing: -0.03em;
        font-size: 2.35rem; line-height: 1.15; margin: 0.8rem 0 0.5rem; padding: 0;
    }
    .hero p {color: #CFE0F5; font-size: 1.04rem; max-width: 640px; margin: 0;}

    /* ---------- Steps ---------- */
    .steps {display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0 0 1.1rem;}
    .step {
        flex: 1 1 200px; display: flex; align-items: center; gap: 0.7rem;
        background: var(--card); border: 1px solid var(--line);
        border-radius: 14px; padding: 0.75rem 1rem;
        font-weight: 600; font-size: 0.92rem; color: var(--navy);
    }
    .step b {
        flex: none; width: 1.8rem; height: 1.8rem; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        background: var(--blue-soft); color: var(--blue); font-size: 0.85rem;
    }
    .step:nth-child(3) b {background: var(--amber-soft); color: #B36B00;}

    /* ---------- Buttons ---------- */
    .stButton > button, .stLinkButton > a, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 999px; font-weight: 600; min-height: 2.7rem;
        transition: all .15s ease;
    }
    .stButton > button, .stLinkButton > a {
        background: #fff; color: var(--blue); border: 1.5px solid var(--blue);
    }
    .stButton > button:hover, .stLinkButton > a:hover {
        background: var(--blue-soft); color: var(--blue-dark); border-color: var(--blue-dark);
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--blue); color: #fff; border: none;
        box-shadow: 0 6px 16px rgba(10, 102, 194, 0.28);
        min-height: 3rem; font-size: 1rem;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--navy); color: #fff;
        box-shadow: 0 8px 20px rgba(11, 37, 69, 0.3);
    }
    div[data-testid="stFormSubmitButton"] > button:active {background: var(--amber); color: var(--navy);}

    /* ---------- Form ---------- */
    div[data-testid="stForm"] {
        background: var(--card); border: 1px solid var(--line);
        border-top: 4px solid var(--blue); border-radius: 16px;
        padding: 1.6rem 1.6rem 1.2rem;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.06);
    }
    label, [data-testid="stWidgetLabel"] p {
        color: var(--navy) !important; font-weight: 600 !important; font-size: 0.9rem !important;
    }
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        border-radius: 10px !important; border-color: var(--line) !important; background: #FBFCFE !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.18) !important;
    }

    /* ---------- Results ---------- */
    .section-title {
        display: flex; align-items: center; gap: 0.6rem;
        font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;
        color: var(--navy); margin: 1.6rem 0 0.8rem;
    }
    .section-title::before {
        content: ""; width: 6px; height: 1.4rem; border-radius: 3px;
        background: linear-gradient(var(--blue), var(--amber));
    }
    .panel {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 16px; padding: 1.3rem 1.4rem; height: 100%;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.05);
    }
    .panel.before {border-top: 4px solid #9AA9BA; background: #FAFBFD;}
    .panel.after {border-top: 4px solid var(--blue);}
    .panel h4 {
        margin: 0 0 0.9rem; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
    }
    .panel.after h4 {color: var(--blue);}
    .panel .label {
        font-size: 0.78rem; font-weight: 700; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.06em; margin: 0.9rem 0 0.3rem;
    }
    .panel .label:first-of-type {margin-top: 0;}
    .panel p {margin: 0 0 0.7rem; line-height: 1.6; font-size: 0.95rem;}
    .panel .muted, .muted {color: var(--muted);}

    .hl-card {
        display: flex; gap: 0.8rem; align-items: flex-start;
        background: var(--blue-soft); border: 1px solid #C9DDF3;
        border-left: 4px solid var(--blue); border-radius: 12px;
        padding: 0.75rem 0.9rem; margin-bottom: 0.6rem;
    }
    .hl-card .num {
        flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%;
        background: var(--amber); color: var(--navy); font-weight: 800; font-size: 0.8rem;
        display: inline-flex; align-items: center; justify-content: center; margin-top: 2px;
    }
    .hl-card span.txt {font-weight: 600; font-size: 0.96rem; line-height: 1.45;}
    .about-box {
        background: var(--amber-soft); border: 1px solid #F7DDAE;
        border-left: 4px solid var(--amber); border-radius: 12px; padding: 0.9rem 1.1rem 0.3rem;
    }

    div[data-testid="stExpander"] {
        background: var(--card); border: 1px solid var(--line) !important; border-radius: 12px;
    }
    div[data-testid="stExpander"] summary {font-weight: 600; color: var(--navy);}
    div[data-testid="stAlert"] {border-radius: 12px;}
    hr {border-color: var(--line);}

    /* ---------- Live output box with green progress outline ---------- */
    .ring {
        position: relative; overflow: hidden; background: var(--card);
        border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem 1.3rem 1.1rem; margin: 1rem 0;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.06);
    }
    .ring .edge {
        position: absolute; display: block; z-index: 2; background: var(--green);
        box-shadow: 0 0 10px rgba(31, 168, 85, 0.75); transition: all .15s linear;
    }
    .ring .edge.top    {top: 0;    left: 0;  height: 4px;}
    .ring .edge.right  {top: 0;    right: 0; width: 4px;}
    .ring .edge.bottom {bottom: 0; right: 0; height: 4px;}
    .ring .edge.left   {bottom: 0; left: 0;  width: 4px;}
    .ring.thinking {animation: ringpulse 1.4s ease-in-out infinite;}
    .ring.done {border-color: var(--green);}
    @keyframes ringpulse {
        0%, 100% {box-shadow: 0 0 0 0 rgba(31, 168, 85, 0), 0 4px 14px rgba(11, 37, 69, 0.06);}
        50% {box-shadow: 0 0 0 5px rgba(31, 168, 85, 0.22), 0 4px 14px rgba(11, 37, 69, 0.06);}
    }
    .ring-head {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.6rem; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .ring-status {color: var(--green); display: inline-flex; align-items: center; gap: 0.5rem;}
    .ring-status::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--green);
    }
    .ring:not(.done) .ring-status::before {animation: ringdot 1s ease-in-out infinite;}
    @keyframes ringdot {50% {opacity: 0.25;}}
    .ring-meta {color: var(--muted); font-variant-numeric: tabular-nums;}
    .ring-body {
        max-height: 320px; overflow-y: auto;
        display: flex; flex-direction: column-reverse;  /* keeps the newest text in view */
    }
    .ring-body p {margin: 0 0 0.6rem; line-height: 1.6; font-size: 0.95rem;}
    .ring-body .live-h {
        margin: 0.4rem 0 0.4rem; font-size: 0.75rem; font-weight: 700; color: var(--blue);
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    .ring-body .cursor {
        display: inline-block; width: 7px; height: 1em; background: var(--amber);
        margin-left: 2px; vertical-align: text-bottom; animation: ringdot 0.8s steps(2) infinite;
    }

    /* ---------- Compare rows, tags, chips, diff ---------- */
    .col-head {
        font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;
        padding: 0.45rem 0.9rem; border-radius: 10px; margin-bottom: 0.2rem;
    }
    .col-head.b {background: #EAEEF3; color: var(--muted);}
    .col-head.a {background: var(--blue-soft); color: var(--blue);}
    .row-label {
        display: flex; align-items: center; gap: 0.6rem; margin: 1.2rem 0 0.5rem;
        font-size: 0.82rem; font-weight: 800; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--navy);
    }
    .row-label::after {content: ""; flex: 1; height: 2px; background: linear-gradient(90deg, var(--amber), transparent);}
    .stats {display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.6rem 0 0.4rem;}
    .stat {
        flex: 1 1 150px; background: var(--card); border: 1px solid var(--line);
        border-radius: 14px; padding: 0.7rem 1rem; border-top: 3px solid var(--blue);
    }
    .stat .v {font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; color: var(--navy);}
    .stat .k {font-size: 0.74rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;}
    .stat.good {border-top-color: var(--green);} .stat.good .v {color: #12793C;}
    .stat.bad {border-top-color: #D92D20;} .stat.bad .v {color: #B42318;}
    .stat.warm {border-top-color: var(--amber);}
    .panel .tag {
        display: inline-block; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--muted); background: #EAEEF3;
        padding: 0.15rem 0.55rem; border-radius: 999px; margin-bottom: 0.7rem;
    }
    .chip {
        display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.75rem;
        border-radius: 999px; font-size: 0.85rem; font-weight: 600;
        margin: 0 0.4rem 0.5rem 0; border: 1px solid;
    }
    .chip.plain {background: #EEF1F5; border-color: var(--line); color: var(--muted);}
    .chip.keep {background: var(--blue-soft); border-color: #C9DDF3; color: var(--blue-dark);}
    .chip.add {background: #E7F7EE; border-color: #A9E0BF; color: #12793C;}
    .chip.add::before {content: "\\2714"; font-weight: 800; color: var(--green);}
    .chip.del {background: #FDECEC; border-color: #F4B8B8; color: #B42318;}
    .chip.del::before {content: "\\2716"; font-weight: 800; color: #D92D20;}
    .chip.del .t {text-decoration: line-through;}
    .legend {font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem;}
    .legend .g {color: #12793C; font-weight: 700;} .legend .r {color: #B42318; font-weight: 700;}
    p.diff {line-height: 1.9; font-size: 0.95rem;}
    ins {background: #E7F7EE; color: #12793C; text-decoration: none; padding: 0 3px; border-radius: 4px;}
    del {background: #FDECEC; color: #B42318; text-decoration: line-through; padding: 0 3px; border-radius: 4px;}

    /* ---------- Footer branding ---------- */
    .brand {
        text-align: center; margin: 2.4rem 0 0.4rem; font-size: 0.8rem;
        color: var(--muted); letter-spacing: 0.02em;
    }
    .brand .dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: var(--amber); margin-right: 0.45rem; vertical-align: middle;
    }
    .brand strong {color: var(--navy); font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="pill">Student edition</span>
        <h1>LinkedIn Profile Optimizer</h1>
        <p>Turn your current profile into three sharp headline ideas and a
        role-focused About section &mdash; honest, specific, and easy to scan.</p>
    </div>
    <div class="steps">
        <div class="step"><b>1</b>Add your profile details</div>
        <div class="step"><b>2</b>Let AI draft the upgrade</div>
        <div class="step"><b>3</b>Review, edit &amp; copy</div>
    </div>
    """,
    unsafe_allow_html=True,
)

action_a, action_b, _spacer = st.columns([1.2, 1.2, 2.6])
with action_a:
    load_sample = st.button(
        "Load next shuffled sample",
        help="Cycles through every example in a shuffled order before reshuffling.",
        use_container_width=True,
    )
with action_b:
    st.link_button(
        "Get a free Gemini API key",
        "https://aistudio.google.com/apikey",
        use_container_width=True,
    )

if load_sample:
    sample_path = Path(__file__).resolve().with_name("sample_profiles.json")
    try:
        sample_profiles = json.loads(sample_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load sample profiles: {exc}")
    else:
        if not isinstance(sample_profiles, list) or not sample_profiles:
            st.error("The sample profiles file must contain a non-empty JSON list.")
        else:
            sample_order = st.session_state.get("_sample_profile_order", [])
            next_sample = st.session_state.get("_next_sample_index", 0)
            if not sample_order or next_sample >= len(sample_order):
                sample_order = list(range(len(sample_profiles)))
                random.shuffle(sample_order)
                previous_sample = st.session_state.get("_last_sample_index")
                if len(sample_order) > 1 and sample_order[0] == previous_sample:
                    sample_order[0], sample_order[1] = (
                        sample_order[1],
                        sample_order[0],
                    )
                next_sample = 0

            sample_index = sample_order[next_sample]
            sample_profile = sample_profiles[sample_index]
            st.session_state["_sample_profile_order"] = sample_order
            st.session_state["_next_sample_index"] = next_sample + 1
            st.session_state["_last_sample_index"] = sample_index
            st.session_state["_sample_scenario"] = sample_profile["scenario"]
            st.session_state["headline"] = sample_profile["current_headline"]
            st.session_state["about"] = sample_profile["current_about"]
            st.session_state["skills"] = sample_profile["skills"]
            st.session_state["target_role"] = sample_profile["target_role"]
            st.session_state.pop("optimized_profile", None)
            st.session_state.pop("original_profile", None)
            st.success(f"Loaded sample: {sample_profile['scenario']}")
            log.info("Loaded sample #%d: %s", sample_index, sample_profile["scenario"])

st.caption("The examples are fictional. Click the button to load another profile.")

# --------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------
with st.form("profile_form"):
    headline_column, role_column = st.columns(2)
    with headline_column:
        current_headline = st.text_input(
            "Current LinkedIn headline",
            placeholder="Student | Aspiring data analyst",
            key="headline",
        )
    with role_column:
        target_role = st.text_input(
            "Target role",
            placeholder="Junior data analyst",
            key="target_role",
        )
    current_about = st.text_area(
        "Current About section",
        height=150,
        placeholder="Paste your current LinkedIn About section...",
        key="about",
    )
    skills = st.text_area(
        "Skills",
        height=100,
        placeholder="Python, data analysis, communication",
        key="skills",
    )
    submitted = st.form_submit_button(
        "Optimize my profile",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------
if submitted:
    st.session_state.pop("optimized_profile", None)
    st.session_state.pop("original_profile", None)
    missing = [
        label
        for label, value in (
            ("current headline", current_headline),
            ("About section", current_about),
            ("skills", skills),
            ("target role", target_role),
        )
        if not value.strip()
    ]
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY in .env")
    if not GEMINI_MODEL:
        missing.append("GEMINI_MODEL in .env")
    if missing:
        log.warning("Submit blocked, missing: %s", ", ".join(missing))
        st.error("Please provide: " + ", ".join(missing) + ".")
    else:
        log.info(
            "Optimize clicked | headline=%d chars | about=%d chars | skills=%d chars | role=%r",
            len(current_headline),
            len(current_about),
            len(skills),
            target_role.strip(),
        )
        live = st.empty()
        text = ""
        expected = st.session_state.get("_expected_chars", DEFAULT_EXPECTED_CHARS)
        started = time.perf_counter()
        try:
            live.markdown(live_box("", 0.0, 0.0), unsafe_allow_html=True)
            for piece in stream_profile(
                api_key=GEMINI_API_KEY,
                model=GEMINI_MODEL,
                current_headline=current_headline.strip(),
                current_about=current_about.strip(),
                skills=skills.strip(),
                target_role=target_role.strip(),
            ):
                text += piece
                progress = min(len(text) / expected, 0.97)  # never claims 100% early
                live.markdown(
                    live_box(text, progress, time.perf_counter() - started),
                    unsafe_allow_html=True,
                )
            optimized_profile = text.strip()
            live.markdown(
                live_box(text, 1.0, time.perf_counter() - started, done=True),
                unsafe_allow_html=True,
            )
            time.sleep(0.7)  # let the finished green outline show
            # Learn the real answer length so the next run's progress is more accurate.
            st.session_state["_expected_chars"] = max(
                300, int((expected + len(optimized_profile)) / 2)
            )
            st.session_state["optimized_profile"] = optimized_profile
            st.session_state["original_profile"] = {
                "headline": current_headline.strip(),
                "about": current_about.strip(),
                "skills": skills.strip(),
                "target_role": target_role.strip(),
            }
        except (errors.APIError, RuntimeError) as exc:
            if getattr(exc, "code", None) in (429, 503):
                st.warning(
                    "Gemini is busy right now (high demand or a free-tier limit). "
                    "I retried automatically but it did not clear. Please try again "
                    "in a minute, or set GEMINI_FALLBACK_MODEL in .env."
                )
            else:
                st.error(
                    "Gemini request failed "
                    f"({type(exc).__name__}). Check the .env configuration and "
                    "API access, then try again. Details are in the terminal log."
                )
        finally:
            live.empty()

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if "optimized_profile" in st.session_state:
    original = st.session_state["original_profile"]
    result_text = st.session_state["optimized_profile"]
    parsed = parse_result(result_text)
    headlines, about_text, suggested = parsed["headlines"], parsed["about"], parsed["skills"]

    title_col, copy_all_col = st.columns([4, 1.3])
    with title_col:
        st.markdown(
            "<div class='section-title' style='margin-top:1.2rem'>Your upgrade</div>",
            unsafe_allow_html=True,
        )

    if not (headlines and about_text):
        # Model ignored the format: show it as-is rather than hiding it.
        log.warning("Could not parse headlines/About from the response; showing raw text")
        with copy_all_col:
            copy_button(result_text, "Copy all")
        st.markdown(result_text)
    else:
        diff = skill_diff(original["skills"], suggested) if suggested else None
        full_text = (
            "Headline options:\n"
            + "\n".join(f"{i}. {h}" for i, h in enumerate(headlines, 1))
            + f"\n\nAbout:\n{about_text}"
        )
        if diff:
            final_skills = diff["kept"] + diff["added"]
            full_text += "\n\nSkills:\n" + ", ".join(final_skills)
        with copy_all_col:
            copy_button(full_text, "Copy everything")

        st.markdown(
            f"<div class='muted' style='margin:-0.4rem 0 0.4rem'>Optimized for "
            f"<strong>{html.escape(original['target_role'])}</strong></div>",
            unsafe_allow_html=True,
        )

        words_before, words_after = len(original["about"].split()), len(about_text.split())
        tiles = (
            f"<div class='stat'><div class='v'>{len(headlines)}</div><div class='k'>Headline options</div></div>"
            f"<div class='stat warm'><div class='v'>{words_before} &rarr; {words_after}</div><div class='k'>About words</div></div>"
        )
        if diff:
            tiles += (
                f"<div class='stat good'><div class='v'>+{len(diff['added'])}</div><div class='k'>Skills added</div></div>"
                f"<div class='stat bad'><div class='v'>&minus;{len(diff['removed'])}</div><div class='k'>Skills removed</div></div>"
            )
        st.markdown(f"<div class='stats'>{tiles}</div>", unsafe_allow_html=True)

        LEFT, RIGHT = 1, 1.7
        head_l, head_r = st.columns([LEFT, RIGHT], gap="medium")
        head_l.markdown("<div class='col-head b'>Before</div>", unsafe_allow_html=True)
        head_r.markdown("<div class='col-head a'>Suggested</div>", unsafe_allow_html=True)

        # ---- Row 1: headline ----
        st.markdown("<div class='row-label'>Headline</div>", unsafe_allow_html=True)
        left, right = st.columns([LEFT, RIGHT], gap="medium")
        left.markdown(
            f"<div class='panel before'><span class='tag'>Current</span>{text_block(original['headline'])}</div>",
            unsafe_allow_html=True,
        )
        with right:
            for i, headline in enumerate(headlines, start=1):
                card_col, btn_col = st.columns([5, 1.3], gap="small")
                card_col.markdown(
                    f'<div class="hl-card"><span class="num">{i}</span>'
                    f'<span class="txt">{html.escape(headline)}</span></div>',
                    unsafe_allow_html=True,
                )
                with btn_col:
                    copy_button(headline, "Copy")

        # ---- Row 2: About ----
        st.markdown("<div class='row-label'>About section</div>", unsafe_allow_html=True)
        left, right = st.columns([LEFT, RIGHT], gap="medium")
        left.markdown(
            f"<div class='panel before'><span class='tag'>Current</span>{text_block(original['about'])}</div>",
            unsafe_allow_html=True,
        )
        with right:
            card_col, btn_col = st.columns([5, 1.3], gap="small")
            card_col.markdown(
                f"<div class='about-box'>{text_block(about_text)}</div>",
                unsafe_allow_html=True,
            )
            with btn_col:
                copy_button(about_text, "Copy")
            with st.expander("See exact word changes"):
                st.markdown(
                    word_diff_html(original["about"], about_text)
                    + "<div class='legend'><span class='g'>Green</span> = added &nbsp;&middot;&nbsp; "
                    "<span class='r'>Red</span> = removed</div>",
                    unsafe_allow_html=True,
                )

        # ---- Row 3: Skills ----
        st.markdown("<div class='row-label'>Skills</div>", unsafe_allow_html=True)
        left, right = st.columns([LEFT, RIGHT], gap="medium")
        left.markdown(
            "<div class='panel before'><span class='tag'>Current</span>"
            f"{chips(split_skills(original['skills']), 'plain') or '<span class=muted>No skills listed.</span>'}</div>",
            unsafe_allow_html=True,
        )
        with right:
            if diff:
                card_col, btn_col = st.columns([5, 1.3], gap="small")
                card_col.markdown(
                    "<div class='panel after'>"
                    f"{chips(diff['kept'], 'keep')}{chips(diff['added'], 'add')}{chips(diff['removed'], 'del')}"
                    "<div class='legend'><span class='g'>&#10004; Added</span> &nbsp;&middot;&nbsp; "
                    "<span class='r'>&#10006; Removed</span> &nbsp;&middot;&nbsp; Blue = kept</div></div>",
                    unsafe_allow_html=True,
                )
                with btn_col:
                    copy_button(", ".join(final_skills), "Copy")
            else:
                log.warning("No skills section found in the response")
                st.info("Gemini did not return a skills section this time. Try again.")

    st.caption(
        "Review the suggestions before using them. Keep only claims and skills "
        "that accurately reflect your experience."
    )

st.markdown(
    f"<div class='brand'><span class='dot'></span>Made by <strong>{CREATOR}</strong></div>",
    unsafe_allow_html=True,
)
