import streamlit as st
import faiss, pandas as pd, torch
from sentence_transformers import SentenceTransformer

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_DTYPE = torch.float32
ABSTRACT_CHARS = 600
MAX_NEW_TOKENS = 300

st.set_page_config(page_title="arXiv RAG", page_icon="📚")
st.title("📚 arXiv RAG: Semantic Search + LLM")
st.caption("50k recent cs.CL / cs.LG / cs.AI / cs.IR abstracts")


@st.cache_resource
def load_search():
    df = pd.read_parquet("papers.parquet")
    index = faiss.read_index("arxiv.index")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return df, index, embedder


@st.cache_resource
def load_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=LLM_DTYPE)
    llm.eval()
    return tok, llm


df, index, embedder = load_search()


def search(q, k):
    v = embedder.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(v, k)
    hits = df.iloc[ids[0]].copy()
    hits["score"] = scores[0]
    return hits


def generate(question, hits):
    tok, llm = load_llm()

    ctx = "\n\n".join(
        f"[{i+1}] {r.title}\n{r.abstract[:ABSTRACT_CHARS]}"
        for i, (_, r) in enumerate(hits.iterrows())
    )

    msgs = [
        {"role": "system", "content":
            "You are a research assistant. Using ONLY the numbered abstracts, "
            "write a 3-4 sentence summary that answers the request. "
            "Cite sources like [1], [2] after each claim. "
            "If the abstracts don't contain the answer, say so."},
        {"role": "user", "content":
            f"Abstracts:\n{ctx}\n\n"
            f"Request: Explain what these papers say about: {question}\n\n"
            f"Summary:"},
    ]

    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = llm.generate(
            **x,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


question = st.text_input("Ask about ML / NLP research",
                         placeholder="e.g. What methods exist to reduce hallucinations in LLMs?")
k = st.slider("Papers to retrieve", 2, 8, 4)
use_llm = st.checkbox("Generate an LLM answer (slower, uses more memory)")

if question:
    hits = search(question, k)

    if use_llm:
        with st.spinner("Generating answer... (can take 30-60s on free CPU)"):
            st.subheader("Answer")
            st.write(generate(question, hits))

    st.subheader("Top papers")
    for i, (_, r) in enumerate(hits.iterrows()):
        with st.expander(f"[{i+1}] {r.title}  (score {r.score:.2f})"):
            st.write(r.abstract)
            st.markdown(f"[arXiv:{r.id}](https://arxiv.org/abs/{r.id})")
