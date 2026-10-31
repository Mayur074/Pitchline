"""
Generates natural-language match previews from model output probabilities
using the Claude API.

Takes a raw probability distribution (home win / draw / away win) plus
supporting features and produces an analyst-style preview that explains
*why* the model leans the way it does, rather than just restating numbers.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to a .env file or your environment."
            )
        _client = Anthropic(api_key=api_key)
    return _client


PREVIEW_SYSTEM_PROMPT = """You are a football analyst writing brief, sharp match previews \
for a data-driven World Cup prediction dashboard. You are given model output (win/draw/loss \
probabilities and supporting stats), and your job is to explain the prediction in plain \
language, the way a smart analyst would on a broadcast.

Rules:
- 2-3 sentences maximum.
- Lead with the statistical edge, not a generic intro.
- Mention the ELO gap only if it's meaningful (over ~75 points).
- If the match is close to a coin flip (all three outcomes within ~10 points of each other), say so plainly instead of forcing a confident pick.
- No hedging filler like "anything can happen in football" — be specific about what the numbers say.
- Do not invent stats, player names, or storylines not provided to you.
"""


def generate_match_preview(
    home_team: str,
    away_team: str,
    prob_home_win: float,
    prob_draw: float,
    prob_away_win: float,
    elo_diff: float,
) -> str:
    """
    Calls the Claude API to turn raw model probabilities into a short,
    readable match preview.

    Falls back to a templated preview if the API call fails, so the
    pipeline never breaks on a network or auth issue.
    """
    user_prompt = f"""Match: {home_team} vs {away_team}

Model output:
- {home_team} win probability: {prob_home_win:.1%}
- Draw probability: {prob_draw:.1%}
- {away_team} win probability: {prob_away_win:.1%}
- ELO rating differential ({home_team} minus {away_team}): {elo_diff:+.0f}

Write the preview now."""

    try:
        client = _get_client()
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            system=PREVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    except Exception as exc:
        logger.warning("Claude API call failed (%s). Falling back to templated preview.", exc)
        return _fallback_preview(home_team, away_team, prob_home_win, prob_draw, prob_away_win)


def _fallback_preview(
    home_team: str, away_team: str,
    prob_home_win: float, prob_draw: float, prob_away_win: float,
) -> str:
    """Deterministic, no-API-required preview used in tests and as a fallback."""
    probs = {home_team: prob_home_win, "Draw": prob_draw, away_team: prob_away_win}
    favorite = max(probs, key=probs.get)

    if favorite == "Draw":
        return f"This one is too close to call confidently. The model gives a {prob_draw:.0%} chance of a draw, with both sides nearly even otherwise."

    edge = probs[favorite]
    if edge < 0.40:
        return f"A genuinely open match. {favorite} hold a slight edge at {edge:.0%}, but this is close to a coin flip."
    return f"{favorite} are favored at {edge:.0%} to win this fixture, based on current form and rating differential."


if __name__ == "__main__":
    preview = generate_match_preview(
        home_team="Brazil",
        away_team="Argentina",
        prob_home_win=0.42,
        prob_draw=0.28,
        prob_away_win=0.30,
        elo_diff=35,
    )
    print(preview)
