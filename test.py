"""
test.py — Stage S prompt audition harness

Plays both PM_PROMPT_V2 paths through your real production pipeline so you
can hear exactly what Sam will sound like before deploying any tweaks:

  Path A — PM_PROMPT_V2_BASE alone     (no-cache quick reply, 15-25 words)
  Path B — BASE + CACHED_SUFFIX         (follow-up using cached research)

Pipeline mirrored:
  Groq llama-3.3-70b-versatile  →  Cartesia sonic-turbo  →  speakers/MP3

Both paths use the SAME models, SAME temperature, SAME voice ID, and the
SAME key rotation as websocket_server.py and Speaker.py. If the audio
sounds right here, it'll sound right in production.

Usage:
    python test.py                  # runs both paths, plays + saves MP3s
    python test.py --base-only      # only Path A
    python test.py --cached-only    # only Path B
    python test.py --no-play        # save MP3s but don't auto-play
    python test.py --question "..."  # override the test question

Outputs saved to ./tts_test_output/<timestamp>_<path>.mp3 for replay.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Load .env so GROQ_API_KEYS, CARTESIA_API_KEYS, etc. are available exactly
# like they are when server.py runs.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[Test] ⚠️  python-dotenv not installed — relying on shell env vars")

import httpx

# ── Production prompts (the ones we just merged) ─────────────────────────────
try:
    from Agent import PM_PROMPT_V2_BASE, PM_PROMPT_V2_CACHED_SUFFIX
except ImportError as e:
    print(f"[Test] ❌ Cannot import PM_PROMPT_V2 from Agent.py: {e}")
    print("[Test]    Make sure you're running this from the AIBOT_V2_langraph folder.")
    sys.exit(1)

# ── Production key rotator ───────────────────────────────────────────────────
try:
    from key_rotator import load_keys, key_for_request
except ImportError as e:
    print(f"[Test] ❌ Cannot import key_rotator: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — matches Speaker.py and websocket_server.py exactly
# ═══════════════════════════════════════════════════════════════════════════

GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.5

CARTESIA_VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
CARTESIA_MODEL    = "sonic-turbo"
CARTESIA_VERSION  = "2025-04-16"

OUTPUT_DIR = Path("tts_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO DATA — what the cached suffix gets filled with for Path B
# Mirror values you'd realistically see in a live meeting so the audio
# preview is representative.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_QUESTION_BASE = "Sam, can you tell me a bit about yourself?"

DEFAULT_QUESTION_CACHED = "Tell me more about that login fix you mentioned."

CACHED_FIXTURE = {
    "client_profile_block": (
        "COMPANY: Celebal Technologies\n"
        "KEY FACTS BELOW (this describes the speaker's company):\n"
        "- Specializes in Generative AI, Data Engineering, and Salesforce CRM rollouts\n"
        "- Sahil is the primary technical contact\n"
        "- Active engagement: SCRUM-244 login page redesign, ongoing"
    ),
    "agenda_block": "Review SCRUM-244 current status and blockers",
    "cached_tickets": (
        "- SCRUM-244 [In Progress]: Login page redesign and OTP flow\n"
        "- SCRUM-267 [In Progress]: Login page mobile responsiveness\n"
        "- SCRUM-275 [To Do]: Login page accessibility audit"
    ),
    "cached_web": (
        "[1] Salesforce Login Best Practices 2026: implement OTP via SMS or "
        "authenticator app, use rate-limiting on login endpoints, log all "
        "auth attempts for audit, and use Salesforce Identity Verification "
        "for high-risk users.\n\n"
        "[2] OWASP Top 10 — Authentication: validate input server-side, "
        "rotate session tokens on privilege escalation, never accept "
        "credentials over GET, enforce MFA for admin accounts."
    ),
    "cached_synthesis": (
        "Yeah, the login fix on SCRUM-244 is in progress. We're rolling out "
        "OTP via SMS first, then layering in Salesforce Identity Verification "
        "for the higher-risk user segment. Sahil's leading the implementation."
    ),
    "conversation_block": (
        "Sahil: How are we handling login security on SCRUM-244?\n"
        "Sam: Yeah, the login fix on SCRUM-244 is in progress. We're rolling out OTP via SMS first, then layering in Salesforce Identity Verification for the higher-risk user segment. Sahil's leading the implementation.\n"
        "Sahil: Tell me more about that login fix you mentioned."
    ),
    "cache_age_sec": 12,
}


# ═══════════════════════════════════════════════════════════════════════════
# Groq call — same shape as websocket_server.py's Fast PM path
# ═══════════════════════════════════════════════════════════════════════════

async def call_groq(system_prompt: str, user_question: str) -> str:
    """Call Groq llama-3.3-70b with the merged prompt and return the response.

    Uses your real GROQ_API_KEYS via the shared rotator — same key path as
    production, including cooldown awareness.
    """
    # Lazy import so groq isn't required just to run --help
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("[Test] ❌ openai package not installed — `pip install openai`")
        sys.exit(1)

    keys = load_keys("GROQ")
    if not keys:
        print("[Test] ❌ No GROQ_API_KEY(S) configured in env")
        sys.exit(1)

    api_key = key_for_request("GROQ") or keys[0]

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    print(f"[Test]   Model:        {GROQ_MODEL}")
    print(f"[Test]   Temperature:  {GROQ_TEMPERATURE}")
    print(f"[Test]   System chars: {len(system_prompt)}")
    print(f"[Test]   User input:   \"{user_question}\"")
    print(f"[Test]   Calling Groq...")

    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_question},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=300,
        )
    except Exception as e:
        print(f"[Test] ❌ Groq error: {e}")
        sys.exit(1)
    finally:
        try:
            await client.close()
        except Exception:
            pass

    latency_ms = (time.time() - t0) * 1000
    text = response.choices[0].message.content.strip()
    word_count = len(text.split())

    print(f"[Test]   Groq response: {latency_ms:.0f}ms, {word_count} words")
    print(f"[Test]   ────────────────────────────────────────────────────────")
    print(f"[Test]   {text}")
    print(f"[Test]   ────────────────────────────────────────────────────────")

    return text


# ═══════════════════════════════════════════════════════════════════════════
# Cartesia call — same endpoint and params as Speaker.py warmup
# ═══════════════════════════════════════════════════════════════════════════

async def call_cartesia(text: str, output_path: Path) -> bool:
    """Synthesize text via Cartesia sonic-turbo. Saves MP3 to output_path.

    Uses CARTESIA_API_KEYS via the shared rotator. Tries each key on 4xx
    failures so a dead key (like your current key #1 with the 402) doesn't
    block the test.
    """
    keys = load_keys("CARTESIA")
    if not keys:
        print("[Test] ❌ No CARTESIA_API_KEY(S) configured in env")
        return False

    payload = {
        "model_id": CARTESIA_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
        "language": "en",
        "output_format": {
            "container": "mp3",
            "sample_rate": 44100,
            "bit_rate": 192000,
        },
    }

    print(f"[Test]   Cartesia voice: {CARTESIA_VOICE_ID}")
    print(f"[Test]   Cartesia model: {CARTESIA_MODEL}")

    async with httpx.AsyncClient(timeout=30) as client:
        # Try each configured key in turn — match Speaker.py's resilience
        for attempt, key in enumerate(keys, 1):
            headers = {
                "Authorization":   f"Bearer {key}",
                "Cartesia-Version": CARTESIA_VERSION,
                "Content-Type":     "application/json",
            }
            try:
                t0 = time.time()
                resp = await client.post(
                    "https://api.cartesia.ai/tts/bytes",
                    headers=headers,
                    json=payload,
                )
                ms = (time.time() - t0) * 1000
                if resp.status_code in (200, 201):
                    output_path.write_bytes(resp.content)
                    size_kb = len(resp.content) / 1024
                    print(f"[Test]   Cartesia: key #{attempt}/{len(keys)} ✅ "
                          f"{ms:.0f}ms, {size_kb:.1f} KB")
                    print(f"[Test]   Saved:    {output_path}")
                    return True
                else:
                    print(f"[Test]   Cartesia: key #{attempt}/{len(keys)} ❌ "
                          f"{resp.status_code} — {resp.text[:120]}")
            except Exception as e:
                print(f"[Test]   Cartesia: key #{attempt}/{len(keys)} ❌ {e}")

    print(f"[Test] ❌ All {len(keys)} Cartesia key(s) failed")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Auto-play — best-effort, won't block the test if it fails
# ═══════════════════════════════════════════════════════════════════════════

def play_audio(path: Path) -> None:
    """Open the MP3 with the OS default player. Cross-platform best effort."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'afplay "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
        print(f"[Test]   ▶️  Playing in default audio player")
    except Exception as e:
        print(f"[Test]   ⚠️  Could not auto-play ({e}). Open manually: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Path runners
# ═══════════════════════════════════════════════════════════════════════════

async def run_path_a(question: str, play: bool) -> None:
    """Path A — PM_PROMPT_V2_BASE alone. Quick reply, no cached context."""
    print()
    print("═" * 70)
    print("PATH A — PM_PROMPT_V2_BASE alone (no-cache, 15-25 words)")
    print("═" * 70)
    print(f"[Test] Question: \"{question}\"")
    print()

    response_text = await call_groq(PM_PROMPT_V2_BASE, question)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}_path_a_base.mp3"

    print()
    success = await call_cartesia(response_text, out_path)
    if success and play:
        play_audio(out_path)


async def run_path_b(question: str, play: bool) -> None:
    """Path B — BASE + CACHED_SUFFIX. Follow-up using fixture data."""
    print()
    print("═" * 70)
    print("PATH B — PM_PROMPT_V2_BASE + CACHED_SUFFIX (cached, 40-60 words)")
    print("═" * 70)
    print(f"[Test] Question: \"{question}\"")
    print(f"[Test] Cached tickets in fixture: SCRUM-244, SCRUM-267, SCRUM-275")
    print(f"[Test] Cache age: {CACHED_FIXTURE['cache_age_sec']}s")
    print()

    # Build the system prompt the same way websocket_server.py does
    template = PM_PROMPT_V2_BASE + "\n\n" + PM_PROMPT_V2_CACHED_SUFFIX
    system_prompt = template.format(
        client_profile_block=CACHED_FIXTURE["client_profile_block"],
        agenda_block=CACHED_FIXTURE["agenda_block"],
        cached_tickets=CACHED_FIXTURE["cached_tickets"],
        cached_web=CACHED_FIXTURE["cached_web"],
        cached_synthesis=CACHED_FIXTURE["cached_synthesis"],
        conversation_block=CACHED_FIXTURE["conversation_block"],
        question=question,
        cache_age_sec=CACHED_FIXTURE["cache_age_sec"],
    )

    response_text = await call_groq(system_prompt, question)

    # Catch the ESCALATE escape hatch — this is what production does too
    if response_text.strip().upper() == "ESCALATE":
        print()
        print("[Test] ⚠️  Model returned ESCALATE — cache couldn't answer this question.")
        print("[Test]    In production this triggers a fresh research turn.")
        print("[Test]    Try a question that's actually a follow-up to the cached context.")
        return

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{timestamp}_path_b_cached.mp3"

    print()
    success = await call_cartesia(response_text, out_path)
    if success and play:
        play_audio(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage S prompt audition — Groq + Cartesia round-trip test"
    )
    p.add_argument("--base-only", action="store_true",
                   help="Run only Path A (no-cache base prompt)")
    p.add_argument("--cached-only", action="store_true",
                   help="Run only Path B (cached-context suffix)")
    p.add_argument("--no-play", action="store_true",
                   help="Save MP3 but don't auto-play")
    p.add_argument("--question", type=str, default=None,
                   help="Override the test question (applies to whichever path runs)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    play = not args.no_play

    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║             Stage S prompt audition — PM_PROMPT_V2                 ║")
    print("║   Groq llama-3.3-70b → Cartesia sonic-turbo → MP3 (production)     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"[Test] Output dir: {OUTPUT_DIR.resolve()}")
    print(f"[Test] Auto-play:  {'ON' if play else 'OFF (--no-play)'}")

    # Sanity: confirm the prompts are real and look right
    print()
    print(f"[Test] PM_PROMPT_V2_BASE:           {len(PM_PROMPT_V2_BASE)} chars")
    print(f"[Test] PM_PROMPT_V2_CACHED_SUFFIX:  {len(PM_PROMPT_V2_CACHED_SUFFIX)} chars")

    if args.cached_only:
        question = args.question or DEFAULT_QUESTION_CACHED
        await run_path_b(question, play)
    elif args.base_only:
        question = args.question or DEFAULT_QUESTION_BASE
        await run_path_a(question, play)
    else:
        # Both paths, back to back. If --question is given, it applies to both.
        await run_path_a(args.question or DEFAULT_QUESTION_BASE, play)
        await asyncio.sleep(0.5)  # tiny breath between paths
        await run_path_b(args.question or DEFAULT_QUESTION_CACHED, play)

    print()
    print("═" * 70)
    print(f"[Test] Done. MP3s saved in: {OUTPUT_DIR.resolve()}")
    print("═" * 70)


if __name__ == "__main__":
    asyncio.run(main())