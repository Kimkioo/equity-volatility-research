"""Optional AI-assisted research brief generator.

Reads the structured outputs produced by run_pipeline.py
(reports/run_summary.json, reports/model_results.csv) and writes a short
natural-language brief to reports/research_brief.md.

Two modes:
  * Template mode (default, no API key needed): a deterministic summary
    assembled directly from the measured metrics.
  * LLM mode (optional): if the OPENAI_API_KEY environment variable is set
    and the `openai` package is installed, the same structured metrics are
    sent to a language model to draft a more fluent narrative. The LLM never
    sees raw market data and never produces forecasts — it only rephrases
    numbers that the quantitative pipeline already computed.

Usage:
    python generate_research_brief.py
"""

from __future__ import annotations

import json
import os
import sys

from src.config import REPORTS_DIR

SUMMARY_PATH = REPORTS_DIR / "run_summary.json"
OUTPUT_PATH = REPORTS_DIR / "research_brief.md"


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        sys.exit("run_summary.json not found - run `python run_pipeline.py` first.")
    with open(SUMMARY_PATH) as fh:
        return json.load(fh)


def template_brief(s: dict) -> str:
    """Deterministic brief built only from measured pipeline outputs."""
    results = sorted(s["results"], key=lambda r: r["RMSE"])
    best = results[0]
    naive = min(
        (r for r in results if r["Model"].startswith("Naive")), key=lambda r: r["RMSE"]
    )
    beat = best["RMSE"] < naive["RMSE"]
    top_features = list(s["top_features"])[:5]

    lines = [
        "# Research Brief (auto-generated)",
        "",
        f"**Sample:** {s['observations_model']:,} daily observations, "
        f"{s['date_start']} to {s['date_end']}, {s['n_features']} engineered features.",
        f"**Target:** {s['target']} (annualized 5-day forward realized volatility of SPY).",
        f"**Validation:** chronological 70/15/15 split with {s['n_walk_forward_folds']} "
        "expanding walk-forward folds over the test period.",
        "",
        "## Model ranking (test RMSE, lower is better)",
        "",
        "| Model | MAE | RMSE | R2 | QLIKE |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['Model']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | "
            f"{r['R2']:.3f} | {r['QLIKE']:.3f} |"
        )
    lines += [
        "",
        f"**Best model:** {best['Model']} "
        f"(RMSE {best['RMSE']:.4f}, R2 {best['R2']:.3f}).",
        f"**Best naive benchmark:** {naive['Model']} (RMSE {naive['RMSE']:.4f}).",
        (
            f"The best model {'outperformed' if beat else 'did not outperform'} the "
            f"naive benchmark by {abs(1 - best['RMSE'] / naive['RMSE']):.1%} in RMSE."
        ),
        "",
        "## Most important predictors (XGBoost gain)",
        "",
        ", ".join(f"`{f}`" for f in top_features) + ".",
        "",
        "_Feature importance is predictive, not causal._",
    ]
    return "\n".join(lines)


def llm_brief(s: dict) -> str | None:
    """Optional fluent narrative via OpenAI. Returns None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed; falling back to template brief.")
        return None
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a quantitative research assistant. Write a concise one-page "
        "markdown research brief based ONLY on the following measured results "
        "from a volatility forecasting study. Do not invent any numbers or "
        "claims not present in the data.\n\n" + json.dumps(s, indent=2)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main() -> None:
    summary = load_summary()
    brief = llm_brief(summary) or template_brief(summary)
    OUTPUT_PATH.write_text(brief)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
