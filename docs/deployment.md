# Deployment (Streamlit Community Cloud)

Easiest option: free, connects straight to GitHub, no Docker or server to manage.

## Why this platform
| Platform | Verdict |
|---|---|
| **Streamlit Community Cloud** | Free, one-click from GitHub. About 1 GB RAM guaranteed per app, sleeps after 12 h without visitors, no custom domain, only one *private* app per account. |
| Hugging Face Spaces | 16 GB RAM, but Streamlit now runs as a Docker Space and Docker/Gradio Spaces need a paid plan on personal accounts (as of Nov 2026), so it is no longer the easy free option. |

## Why `requirements.txt` has no TensorFlow
TensorFlow alone needs most of the free 1 GB. The app detects TensorFlow automatically
(`tensorflow_available()` uses `importlib.util.find_spec`, so it is never imported just to check),
and only offers the Keras autoencoder when it is installed. The hosted app therefore offers
Isolation Forest, One-Class SVM and the from-scratch models. For the full set locally or in CI:

    pip install -r requirements-dev.txt

## Steps
1. Push the repo to GitHub (already done). If the repo stays private, authorise Streamlit to read it
   (the free tier allows one private app).
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**.
3. Repository `Perera1325/ai-powered-insider-threat-detection`, branch `main`, main file `app.py`.
4. Under *Advanced settings* choose **Python 3.12** (the default may be a newer version that some
   packages do not support yet).
5. Deploy. The first build installs the dependencies (a few minutes).

## Known limits when hosted
- The sleeping app wakes on the next visit (about 30-60 s).
- The Real-Time tab starts a background process and writes to a temp directory. This should work on a
  normal container but has only been tested locally, so check it once after deploying.
- Data under `data/processed/` is committed to the repo, so no generation step is needed at start-up.
