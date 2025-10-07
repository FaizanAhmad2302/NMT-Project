# app.py
import streamlit as st
import tempfile, subprocess, sys, re, os
from pathlib import Path

st.set_page_config(page_title="Urdu → Roman Urdu NMT", layout="centered")

st.title("Urdu → Roman Urdu Translator (NMT)")
st.caption("Uses your repo's seq2seq (BiLSTM + Luong attention) model & checkpoint.")

# sensible defaults from your repo layout
DEFAULT_CKPT = "runs/base/best.pt"
DEFAULT_INPUT_TXT = "urdu_input.txt"

# Sidebar config
st.sidebar.header("Configuration")
ckpt_path = st.sidebar.text_input("Checkpoint (.pt)", DEFAULT_CKPT)
extra_args = st.sidebar.text_input("Extra args for predict.py (optional)", "")

# Input
st.subheader("Enter Urdu text")
text = st.text_area(
    "One sentence per line:",
    height=160,
    placeholder="یہاں اردو لکھیں…"
)

uploaded = st.file_uploader("…or upload a .txt file (one line per sentence)", type=["txt"])

# Utilities
def run_predict(lines):
    """Write lines to temp file, call your predict.py, parse 'Roman Urdu:' lines."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        for ln in lines:
            ln = ln.strip()
            if ln:
                tmp.write(ln + "\n")
        tmp_path = tmp.name

    cmd = [sys.executable, "src/predict.py", "--ckpt", ckpt_path, "--input_file", tmp_path]
    if extra_args.strip():
        cmd.extend(extra_args.split())

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = out.stdout
    except subprocess.CalledProcessError as e:
        st.error("Prediction failed. See details below.")
        st.code(e.stdout + "\n" + e.stderr)
        return []

    # Parse the format shown in README:
    # Urdu: ...
    # Roman Urdu: ...
    roman = []
    for line in stdout.splitlines():
        m = re.match(r"\s*Roman Urdu:\s*(.*)", line)
        if m:
            roman.append(m.group(1).strip())
    # fallback: if nothing matched, show raw for debugging
    if not roman:
        st.warning("Couldn’t parse translations from stdout; showing raw output.")
        st.code(stdout)
    # cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return roman

# Action
if st.button("Translate", type="primary"):
    # Gather lines from either the textarea or upload
    lines = []
    if uploaded is not None:
        lines = uploaded.read().decode("utf-8").splitlines()
    else:
        lines = text.splitlines()

    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        st.info("Please enter some Urdu text or upload a .txt file.")
    elif not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found at: {ckpt_path}")
    else:
        with st.spinner("Translating…"):
            roman_lines = run_predict(lines)

        if roman_lines:
            st.subheader("Results")
            for ur, ro in zip(lines, roman_lines):
                st.markdown(f"**Urdu:** {ur}")
                st.markdown(f"**Roman:** {ro}")
                st.markdown("---")
