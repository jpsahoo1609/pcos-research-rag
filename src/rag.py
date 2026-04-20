from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a medical research assistant specializing in PCOS (Polycystic Ovary Syndrome).
Answer using ONLY the provided context. Cite sources using [PMC ID].
If context is insufficient, say so. Never give medical advice.
Distinguish between established findings and preliminary results.
Note sample sizes and study limitations when relevant."""


def ask(question, searcher, top_k=5):
    results = searcher.search(question, top_k=top_k)

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    context = "\n\n".join([
        f"[{r['metadata']['pmcid']}] ({r['metadata']['section']}): {r['content']}"
        for r in results
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [{
            "pmcid": r["metadata"]["pmcid"],
            "section": r["metadata"]["section"],
            "url": r["metadata"]["source"]
        } for r in results]
    }
