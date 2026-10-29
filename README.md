# Financial NLP Lab

Indonesian Financial News Sentiment Intelligence. This Next.js research interface calls a separate FastAPI service for lazy, on-demand inference using fine-tuned IndoBERT, IndoLEM, and FinBERT checkpoints. The frontend contains no weights or Hugging Face credentials. News text stays in browser memory and is sent only to the configured backend when Analyze is pressed; the API does not persist requests.

## Research artifacts inspected

The neighboring research workspace (`../models`) contains three local fine-tuned checkpoints. Each `config.json` explicitly maps `0 = Negatif`, `1 = Netral`, `2 = Positif`; inference verifies that mapping rather than assuming it. Tokenizers are loaded from their corresponding checkpoints. Training used 5 epochs, learning rate `1e-5`, batch size 16, max length 256, label smoothing 0.1, warmup 0.1, and weight decay 0.01. Inference applies a 256-token truncation limit; training parameters are descriptive only.

`Phase3_Preprocessing_Translation.ipynb` uses `deep-translator`'s `GoogleTranslator(source="id", target="en")`, preserves Indonesian finance terms, caps input at 4,500 characters, and retries four times. The backend follows this adapter and fails FinBERT independently when translation fails. No other translation service is silently substituted. `results/full_vs_gold_gap.csv` reports IndoBERT-FT Full Validation Macro F1 0.8628 vs Gold Validation 0.5035 and IndoLEM-FT 0.8432 vs 0.5120. These are previously reported research metrics, not live inference results; the gap is a central limitation.

## Requirements

- Node.js 20 or newer and npm.
- Python 3.11 recommended for backend and Docker. CPU-only inference is the default; CUDA is selected when available.
- Disk space for dependencies and model downloads. Each discovered local weight file is about 0.4–0.5 GB; allow additional room for tokenizer/cache files and Python packages.

## Local setup

Run commands from the project root. The current workspace is nested alongside the research assets, so `backend/app/config.py` detects `../models/{indobert,indolem,finbert}_finetuned` automatically when present. Those files stay outside this project and must not be copied into the frontend or committed. Otherwise, set each model variable to a local path or Hub repository ID.

### 1. Frontend dependencies

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

The frontend is available at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` in `.env.local` to the backend origin. No `NEXT_PUBLIC_` value may contain `HF_TOKEN`.

### 2. Backend dependencies and environment

PowerShell:

```powershell
py -3.11 -m venv backend/.venv
backend/.venv/Scripts/Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
```

macOS/Linux:

```bash
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

The application reads environment variables from the process; it does not automatically parse `.env`. Export the values from `backend/.env`, or use your shell’s environment-file tooling. Local model folders are detected automatically when present. For Hub downloads, use the default `Tipuls/...` IDs only after confirming the repositories exist and contain these fine-tuned checkpoints. The suggestions have not been verified as available; a load failure is returned as an error and never falls back to an unrelated base model.

Example PowerShell configuration using local paths:

```powershell
$env:INDOBERT_MODEL_ID = "D:\Collage\BerKol\models\indobert_finetuned"
$env:INDOLEM_MODEL_ID = "D:\Collage\BerKol\models\indolem_finetuned"
$env:FINBERT_MODEL_ID = "D:\Collage\BerKol\models\finbert_finetuned"
$env:CORS_ORIGINS = "http://localhost:3000"
python -m uvicorn app.main:app --reload --app-dir backend
```

The API is at `http://localhost:8000`; docs at `/docs`. Check backend health without loading weights using `GET /health`; inspect per-model `configured`, `downloaded`, `loaded`, and `ready` states with `GET /models`.

### 3. Test an individual model

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"text":"Harga saham naik setelah laba perusahaan tumbuh.","model_id":"indobert"}'
```

The first request downloads/loads its checkpoint and can be slow. `MAX_CACHED_MODELS=1` keeps only one full model in memory by default. Comparison runs models sequentially and retains successful outputs if another model or translation fails. Set `MAX_CACHED_MODELS` higher only when backend memory allows it.

### 4. Run checks

```powershell
npm run typecheck
npm run lint
npm run build
python -m unittest discover -s backend/tests -v
python -m compileall -q backend/app backend/tests
```

These checks do not download checkpoints or claim live model inference. For a live smoke check, call `/predict` using a configured local checkpoint after starting the API.

## Configuration reference

Backend variables:

- `INDOBERT_MODEL_ID`, `INDOLEM_MODEL_ID`, `FINBERT_MODEL_ID`: local directories or Hugging Face Hub repository IDs. Defaults are the requested `Tipuls/...` values when adjacent local folders are absent.
- `HF_TOKEN`: backend-only secret for private Hub repositories. Never put this in frontend environment variables or source control.
- `HF_HOME`: Hugging Face model download/cache directory.
- `MAX_CACHED_MODELS`: LRU model count; default 1.
- `MAX_TEXT_CHARS`: API limit; default 8,000.
- `TRANSLATION_PROVIDER`: `google` selects the same `deep-translator` GoogleTranslator adapter as the research notebook. Other values produce an explicit error.
- `CORS_ORIGINS`: comma-separated exact frontend origins; set this to the deployed Vercel origin in production.

Frontend variable:

- `NEXT_PUBLIC_API_URL`: public backend origin, e.g. `https://api.example.com`.

The translated FinBERT input is returned for display. Translation service availability, rate limits, network access, and changed translation output can all affect results. Do not use this tool as financial advice.

## Deployment

### Frontend on Vercel

1. Import this project directory into Vercel.
2. Set `NEXT_PUBLIC_API_URL` to the final backend HTTPS origin.
3. Deploy the Next.js app. The frontend can build and deploy independently; it displays the backend as unavailable until reachable.

### FastAPI backend / Hugging Face Spaces Docker

The `backend/Dockerfile` and `backend/README.md` configure a Docker Space on port 7860. Build context should be the `backend` directory (`docker build -t financial-nlp-api backend`; run with `docker run -p 7860:7860 --env-file backend/.env financial-nlp-api`). On Hugging Face, create a Docker Space with `backend` contents at the Space root, configure model IDs and `CORS_ORIGINS` as Space variables, and add `HF_TOKEN` only as a Space secret if needed. Use a persistent volume/cache if available so model downloads survive restarts.

Hugging Face’s current Spaces documentation lists CPU Basic at 2 vCPU, 16 GB RAM, and 50 GB non-persistent disk; compute Spaces may require a paid plan, and free CPU Spaces can sleep when idle. The three local checkpoint files total roughly 1.38 GB, so sequential one-model caching fits RAM in principle, but cold downloads, startup latency, CPU throughput, disk persistence, translation network limits, and provider rate limits still affect reliability. Confirm current plan and hardware in the account before choosing it. The default Docker image downloads a CPU PyTorch wheel; the weights are fetched at runtime and are not bundled.

If the available tier is too slow or lacks disk, use a persistent volume, a larger CPU/GPU tier, or a compatible Python host with attached storage. For an even tighter RAM limit, retain one-model caching and run each model as a separate service; expose comparison through sequential backend requests. Never add checkpoint weights to Git or a Vercel bundle.

### Production CORS and secrets

Set `CORS_ORIGINS` to the exact production frontend origin (for example `https://your-app.vercel.app`), `NEXT_PUBLIC_API_URL` to the backend HTTPS base URL, and any private `HF_TOKEN` only in the backend secret manager. Restart/redeploy both services after changing environment values. Verify `/health`, `/models`, and one real `/predict` before announcing readiness.

## Known limitations and deployment status

- The suggested Hub repositories could not be verified from the available project artifacts or search results. Confirm the IDs and access policy before deploying; local checkpoints are available outside this project in the current workspace.
- `deep-translator`’s Google provider requires outbound network access and may rate-limit. Translation failure is reported only for FinBERT; other models continue.
- Full Validation and Gold Validation differ substantially; the result UI shows both for IndoBERT-FT and IndoLEM-FT to avoid presenting the lexicon-labeled score as generalization quality.
- No deployment has been performed by this project setup.
