"""
Symptom Code Mapper - Streamlit Application
============================================
Maps investigation summaries from an Excel file to the most relevant
symptom codes using semantic similarity (sentence-transformers).

Usage:
    streamlit run symptom_mapper_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# ── Lazy-load heavy model only when needed ────────────────────────────────────
@st.cache_resource(show_spinner="Loading semantic model…")
def load_model():
    """Load and cache the SentenceTransformer model so it is only downloaded once."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Excel helpers ─────────────────────────────────────────────────────────────

def read_excel_sheets(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read 'Investigation Inputs' and 'Symptom Code' sheets from the uploaded file.

    Returns:
        inv_df  – Investigation Inputs dataframe
        sym_df  – Symptom Code dataframe
    Raises:
        ValueError if expected sheets or columns are missing.
    """
    xls = pd.ExcelFile(uploaded_file)

    required_sheets = {"Investigation Inputs", "Symptom Code"}
    missing = required_sheets - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Missing sheet(s) in uploaded file: {', '.join(missing)}")

    inv_df = xls.parse("Investigation Inputs")
    sym_df = xls.parse("Symptom Code")

    # Normalise column names (strip whitespace)
    inv_df.columns = inv_df.columns.str.strip()
    sym_df.columns = sym_df.columns.str.strip()

    if "Investigation Summary" not in inv_df.columns:
        raise ValueError(
            "Column 'Investigation Summary' not found in the 'Investigation Inputs' sheet."
        )

    return inv_df, sym_df


def detect_symptom_columns(sym_df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect the symptom-code and symptom-description column names.
    Looks for columns whose names contain 'code' or 'description' (case-insensitive).
    Falls back to the first two columns if detection fails.
    """
    cols = sym_df.columns.tolist()

    code_col = next(
        (c for c in cols if "code" in c.lower()),
        cols[0],
    )
    desc_col = next(
        (c for c in cols if "desc" in c.lower() or "name" in c.lower()),
        cols[1] if len(cols) > 1 else cols[0],
    )
    return code_col, desc_col


# ── Semantic similarity core ──────────────────────────────────────────────────

def compute_embeddings(texts: list[str], model) -> np.ndarray:
    """Encode a list of strings into dense embedding vectors."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two sets of vectors.

    Args:
        a: shape (m, d)
        b: shape (n, d)
    Returns:
        similarity matrix of shape (m, n)
    """
    # Normalise rows to unit length
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def map_summaries_to_symptoms(
    summaries: list[str],
    symptom_codes: list[str],
    symptom_descriptions: list[str],
    model,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """
    For each investigation summary find the best-matching symptom code.

    Args:
        summaries            – list of investigation summary strings
        symptom_codes        – list of symptom code identifiers
        symptom_descriptions – list of human-readable descriptions (parallel to codes)
        model                – loaded SentenceTransformer
        threshold            – minimum cosine similarity to accept a match
                               (scores below → "Others")
    Returns:
        DataFrame with columns:
            Investigation Summary | Matched Symptom Code |
            Symptom Description   | Confidence Score
    """
    # Build combined text for symptoms: "CODE: description"
    symptom_texts = [
        f"{code}: {desc}"
        for code, desc in zip(symptom_codes, symptom_descriptions)
    ]

    with st.spinner("Computing semantic embeddings…"):
        summary_embeddings  = compute_embeddings(summaries,     model)
        symptom_embeddings  = compute_embeddings(symptom_texts, model)

    sim_matrix = cosine_similarity_matrix(summary_embeddings, symptom_embeddings)

    results = []
    for i, summary in enumerate(summaries):
        best_idx   = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])

        if best_score >= threshold:
            matched_code = symptom_codes[best_idx]
            matched_desc = symptom_descriptions[best_idx]
        else:
            matched_code = "Others"
            matched_desc = "No close match found"

        results.append(
            {
                "Investigation Summary": summary,
                "Matched Symptom Code":  matched_code,
                "Symptom Description":   matched_desc,
                "Confidence Score":      round(best_score, 4),
            }
        )

    return pd.DataFrame(results)


# ── Excel export helper ───────────────────────────────────────────────────────

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to an in-memory Excel (.xlsx) byte string."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Mapped Results")
    return buffer.getvalue()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Symptom Code Mapper",
        page_icon="🔬",
        layout="wide",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 Investigation Summary → Symptom Code Mapper")
    st.markdown(
        """
        Upload your Excel workbook (containing **Investigation Inputs** and
        **Symptom Code** sheets) and this tool will automatically map each
        investigation summary to the most semantically similar symptom code.
        """
    )

    # ── Sidebar settings ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            label="Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help=(
                "Summaries whose best match scores below this value are "
                "labelled 'Others'."
            ),
        )
        st.markdown("---")
        st.markdown(
            "**Model:** `all-MiniLM-L6-v2`  \n"
            "Fast, lightweight sentence-transformer suitable for "
            "short-to-medium text matching."
        )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)",
        type=["xlsx"],
        help="The file must contain 'Investigation Inputs' and 'Symptom Code' sheets.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an Excel file to get started.")
        st.stop()

    # ── Read sheets ───────────────────────────────────────────────────────────
    try:
        inv_df, sym_df = read_excel_sheets(uploaded_file)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()

    # ── Preview raw data ──────────────────────────────────────────────────────
    with st.expander("📄 Preview: Investigation Inputs", expanded=False):
        st.dataframe(inv_df, use_container_width=True)

    with st.expander("📋 Preview: Symptom Codes", expanded=False):
        st.dataframe(sym_df, use_container_width=True)

    # ── Detect symptom columns ────────────────────────────────────────────────
    code_col, desc_col = detect_symptom_columns(sym_df)

    st.markdown(
        f"**Detected columns →** Code: `{code_col}` | Description: `{desc_col}`"
    )

    # Allow the user to override auto-detection
    all_sym_cols = sym_df.columns.tolist()
    with st.expander("🔧 Override symptom column selection", expanded=False):
        code_col = st.selectbox("Symptom Code column",   all_sym_cols,
                                index=all_sym_cols.index(code_col))
        desc_col = st.selectbox("Symptom Description column", all_sym_cols,
                                index=all_sym_cols.index(desc_col))

    # ── Extract data vectors ──────────────────────────────────────────────────
    summaries            = inv_df["Investigation Summary"].astype(str).tolist()
    symptom_codes        = sym_df[code_col].astype(str).tolist()
    symptom_descriptions = sym_df[desc_col].astype(str).tolist()

    st.markdown(
        f"**Rows to process:** {len(summaries)} summaries  |  "
        f"**Symptom codes available:** {len(symptom_codes)}"
    )

    # ── Run mapping ───────────────────────────────────────────────────────────
    if st.button("▶️ Run Mapping", type="primary"):
        model = load_model()

        results_df = map_summaries_to_symptoms(
            summaries            = summaries,
            symptom_codes        = symptom_codes,
            symptom_descriptions = symptom_descriptions,
            model                = model,
            threshold            = threshold,
        )

        st.session_state["results_df"] = results_df

    # ── Display results (persists across re-runs) ─────────────────────────────
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]

        st.success("✅ Mapping complete!")

        # ── Summary metrics ───────────────────────────────────────────────────
        total      = len(results_df)
        matched    = (results_df["Matched Symptom Code"] != "Others").sum()
        unmatched  = total - matched
        avg_conf   = results_df["Confidence Score"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Summaries",  total)
        col2.metric("Matched",          matched)
        col3.metric("Unmatched (Others)", unmatched)
        col4.metric("Avg Confidence",   f"{avg_conf:.2%}")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("📊 Mapping Results")

        # Colour-code confidence score column
        styled = results_df.style.background_gradient(
            subset=["Confidence Score"], cmap="RdYlGn", vmin=0, vmax=1
        ).format({"Confidence Score": "{:.4f}"})

        st.dataframe(styled, use_container_width=True, height=450)

        # ── Download button ───────────────────────────────────────────────────
        excel_bytes = dataframe_to_excel_bytes(results_df)
        st.download_button(
            label="⬇️ Download Results as Excel",
            data=excel_bytes,
            file_name="symptom_mapping_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
