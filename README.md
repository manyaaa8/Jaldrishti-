# JalDrishti

AI-assisted urban stormwater decision intelligence — a Streamlit prototype for **SIH 2026**.

Predicts **WHERE** flooding will concentrate, diagnoses **WHY**, recommends **WHAT** to do, and ranks **WHO** should get scarce pumps/drainage teams first — with a fully explainable, rule-based scoring model (not a trained ML model) and a real max-flow / min-cut network bottleneck demonstration.

## Features

- Interactive synthetic city drainage network with 4 zones and 2 pumps
- Rainfall / blockage / siltation / storage-loss scenario sliders
- WHERE → hotspot risk ranking + map
- WHY → transparent cause diagnosis (blockage, siltation, storage loss, pump failure, rainfall)
- WHAT → per-zone recommended action
- WHO FIRST → explainable priority queue (risk + population impact + criticality + capacity scarcity)
- 🔴 Simulate pump failure (P1 / P2 independently) with automatic recalculation
- Before / after priority comparison table showing rank movement
- Network bottleneck / max-flow demonstration with min-cut highlighting on the map
- Human-in-the-loop framing throughout — no autonomous actuation, no fabricated accuracy claims

## Requirements

- Python 3.9+
- See `requirements.txt` (streamlit, pandas, numpy, matplotlib)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy (Streamlit Community Cloud — free)

1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repo/branch, set main file path to `app.py`, click **Deploy**.
4. Wait 1–3 minutes for the build. You'll get a public URL (e.g. `jaldrishti.streamlit.app`).
5. Any future `git push` to the connected branch auto-redeploys.

### Alternatives
- **Hugging Face Spaces** — same Git-push workflow, choose the Streamlit SDK.
- **Render / Railway** — more control, use start command:
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Demo script (for judges)

1. Start with a heavy-rain scenario across the four zones.
2. Show WHERE risk concentrates and WHY (cause diagnosis).
3. Show WHAT is recommended and WHO gets resources first, with the explainable priority formula.
4. Click **Fail P1** in the sidebar.
5. Point to the Before/After table (WHO FIRST tab) and the dashed red bottleneck edge on the map — both update automatically.
6. Close on: this re-plans automatically when conditions change, but the human authority still approves and acts.

## Disclaimer

This is a prototype using synthetic demonstration data. The risk score and cause diagnosis are transparent, rule-based models for decision support — not trained ML models — and no real-world results or accuracy figures are claimed.