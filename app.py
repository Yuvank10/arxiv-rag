import streamlit as st
import faiss, pandas as pd, torch
from sentence_transformers import SentenceTransformer

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
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
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
    ctx = "\n\n".join(f"[{i+1}] {r.title}\n{r.abstract}"
                      for i, (_, r) in enumerate(hits.iterrows()))
    msgs = [
        {"role": "system", "content": "Answer using ONLY the abstracts. Cite like [1]. If unsure, say so."},
        {"role": "user", "content": f"Abstracts:\n{ctx}\n\nQuestion: {question}"},
    ]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = tok(p, return_tensors="pt")
    with torch.no_grad():
        o = llm.generate(**x, max_new_tokens=250, do_sample=False)
    return tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True)

question = st.text_input("Ask about ML / NLP research")
k = st.slider("Papers to retrieve", 2, 8, 4)
use_llm = st.checkbox("Generate an LLM answer (slower, uses more memory)")

if question:
    hits = search(question, k)
    if use_llm:
        with st.spinner("Generating answer..."):
            st.subheader("Answer")
            st.write(generate(question, hits))
    st.subheader("Top papers")
    for i, (_, r) in enumerate(hits.iterrows()):
        with st.expander(f"[{i+1}] {r.title}  (score {r.score:.2f})"):
            st.write(r.abstract)
            st.markdown(f"[arXiv:{r.id}](https://arxiv.org/abs/{r.id})")