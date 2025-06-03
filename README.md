# chatbot
A retrieval-augmented chatbot that answers university catalog questions with 100% accuracy using only the provided PDF file. Built in Colab with FAISS, Sentence Transformers, and a local LLM.

## Streamlit interface
After generating `catalog_embeds.npy`, `chunks.json` and `catalog.index` with the notebook, run:

```bash
streamlit run app.py
```

Then open the provided URL to chat with the catalog.
