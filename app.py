# app.py
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Urdu → Roman Urdu Translator (NMT)", layout="centered")

st.title("Urdu → Roman Urdu Translator (NMT)")
st.caption("Uses your repo's seq2seq (BiLSTM + Luong attention) model & checkpoint.")

# ---- Paths & defaults ----
ROOT = Path(__file__).resolve().parent  # repo root at runtime
DEFAULT_CKPT = "runs/base/best.pt"

# ---- Sidebar config ----
st.sidebar.header("Configuration")
ckpt_path = st.sidebar.text_input("Checkpoint (.pt)", DEFAULT_CKPT)
extra_args = st.sidebar.text_input("Extra args for predict.py (optional)", "")

# ---- Input UI ----
st.subheader("Enter Urdu text")
text = st.text_area(
    "One sentence per line:",
    height=160,
    placeholder="یہاں اردو لکھیں…",
)
uploaded = st.file_uploader("…or upload a .txt file (one line per sentence)", type=["txt"])

# ---- Predictor wrapper ----
def run_predict(lines):
    """Write lines to a temp file, invoke src.predict, parse 'Roman Urdu:' lines."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        for ln in lines:
            ln = ln.strip()
            if ln:
                tmp.write(ln + "\n")
        tmp_path = tmp.name

    cmd = [sys.executable, "-m", "src.predict", "--ckpt", ckpt_path, "--input_file", tmp_path]
    if extra_args.strip():
        cmd.extend(extra_args.split())

    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,                                  # run from repo root
            env={**os.environ, "PYTHONPATH": str(ROOT)}  # make 'src' importable
        )
        stdout = out.stdout
    except subprocess.CalledProcessError as e:
        st.error("Prediction failed. See details below.")
        st.code((e.stdout or "") + "\n" + (e.stderr or ""))
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return []

    # Parse lines that look like: "Roman Urdu: <text>"
    roman = []
    for line in stdout.splitlines():
        m = re.match(r"\s*Roman Urdu:\s*(.*)", line)
        if m:
            roman.append(m.group(1).strip())

    if not roman:
        st.warning("Couldn’t parse translations from stdout; showing raw output.")
        st.code(stdout)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return roman

# ---- Action ----
if st.button("Translate", type="primary"):
    if uploaded is not None:
        lines = uploaded.read().decode("utf-8").splitlines()
    else:
        lines = text.splitlines()

    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        st.info("Please enter some Urdu text or upload a .txt file.")
    elif not (ROOT / ckpt_path).exists():
        st.error(f"Checkpoint not found at: {ckpt_path}")
    else:
        with st.spinner("Translating…"):
            roman_lines = run_predict(lines)

        if roman_lines:
            st.subheader("Results")
            for ur, ro in zip(lines, roman_lines):
                st.markdown(f"*Urdu:* {ur}")
                st.markdown(f"*Roman:* {ro}")
                st.markdown("---")