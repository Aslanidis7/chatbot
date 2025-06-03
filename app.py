import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

# Load embeddings and chunks
EMB_PATH = "catalog_embeds.npy"
CHUNKS_PATH = "chunks.json"
INDEX_PATH = "catalog.index"

@st.cache_resource(show_spinner=False)
def load_resources():
    embeddings = np.load(EMB_PATH)
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    index = faiss.read_index(INDEX_PATH)
    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        max_new_tokens=300,
        repetition_penalty=1.1,
    )
    return embeddings, chunks, index, embed_model, generator

embeddings, chunks, index, embed_model, generate = load_resources()


def ask(question, k=7, min_sim=0.25):
    q_vec = embed_model.encode(question, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec.reshape(1, -1), k)
    top_scores = D[0]
    top_ids = I[0]
    if top_scores[0] < min_sim:
        return "Συγγνώμη 🙏 – ρώτα με κάτι που υπάρχει στον πανεπιστημιακό κατάλογο!\nSorry 🙏 – please ask something from the university catalog!"
    ctx_parts = []
    for score, idx in zip(top_scores, top_ids):
        chunk = chunks[int(idx)]
        ctx_parts.append(f"[p.{chunk['page']}] {chunk['text']}")
    context = "\n".join(ctx_parts)
    prompt = (
        "You are UniCatalogBot. Answer ONLY with information found in the CONTEXT. "
        "List course codes, titles, credits & descriptions verbatim when relevant. "
        "Cite page numbers like (see p. 123). If answer not in context, apologise in Greek & English.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    result = generate(prompt)[0]["generated_text"]
    ans_start = result.find("ANSWER:")
    answer = result[ans_start + len("ANSWER:"):].strip() if ans_start != -1 else result
    return answer

st.title("UniCatalogBot")
question = st.text_input("Ask the catalog:")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = ask(question)
    st.write(response)

