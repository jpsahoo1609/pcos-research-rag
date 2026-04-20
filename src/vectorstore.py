import os
from openai import OpenAI
from supabase import create_client

openai_client = OpenAI()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    response = openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def store_chunks(chunks, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = get_embeddings_batch(texts)

        rows = []
        for chunk, emb in zip(batch, embeddings):
            rows.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": emb
            })

        supabase.table("documents").insert(rows).execute()
        print(f"Stored batch {i // batch_size + 1}/{(len(chunks) // batch_size) + 1}")

    print(f"\n✅ All {len(chunks)} chunks stored in Supabase")


def search(query, top_k=5):
    emb = get_embeddings_batch([query])[0]
    result = supabase.rpc("match_documents", {
        "query_embedding": emb,
        "match_count": top_k,
        "filter": {}
    }).execute()
    return result.data
