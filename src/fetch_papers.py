from Bio import Entrez
import requests
import time
import json
import os

Entrez.email = os.getenv("PUBMED_EMAIL", "your-email@gmail.com")


def search_pubmed(query, max_results=20):
    handle = Entrez.esearch(db="pmc", term=query, retmax=max_results, sort="relevance")
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"]


def fetch_full_text(pmcid):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{pmcid}/unicode"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return []

        data = resp.json()

        if isinstance(data, list):
            documents = []
            for item in data:
                if isinstance(item, dict):
                    documents.extend(item.get("documents", []))
        elif isinstance(data, dict):
            documents = data.get("documents", [])
        else:
            return []

        texts = []
        for doc in documents:
            passages = doc.get("passages", []) if isinstance(doc, dict) else []
            for passage in passages:
                if not isinstance(passage, dict):
                    continue
                text = passage.get("text", "")
                infons = passage.get("infons", {})
                section = infons.get("section_type", "unknown") if isinstance(infons, dict) else "unknown"
                if text and len(text) > 50:
                    texts.append({
                        "text": text,
                        "section": section,
                        "pmcid": f"PMC{pmcid}"
                    })
        return texts
    except Exception as e:
        print(f"Error fetching PMC{pmcid}: {e}")
    return []


def download_papers(disease="PCOS", max_papers_per_query=10):
    queries = [
        f"{disease} polycystic ovary syndrome treatment guidelines",
        f"{disease} insulin resistance metabolic syndrome",
        f"{disease} diagnosis Rotterdam criteria ultrasound",
        f"{disease} infertility ovulation induction IVF",
        f"{disease} hormonal imbalance androgen hyperandrogenism",
        f"{disease} weight management lifestyle intervention diet",
        f"{disease} mental health anxiety depression quality of life"
    ]

    all_papers = []
    seen_ids = set()

    for query in queries:
        print(f"\nSearching: {query}")
        ids = search_pubmed(query, max_results=max_papers_per_query)
        print(f"  Found {len(ids)} papers")
        for pmcid in ids:
            if pmcid not in seen_ids:
                seen_ids.add(pmcid)
                passages = fetch_full_text(pmcid)
                if passages:
                    print(f"  ✅ PMC{pmcid} — {len(passages)} passages")
                    all_papers.extend(passages)
                else:
                    print(f"  ❌ PMC{pmcid} — no full text")
                time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Total: {len(all_papers)} passages from {len(seen_ids)} papers")

    os.makedirs("data", exist_ok=True)
    with open("data/papers.json", "w") as f:
        json.dump(all_papers, f)

    return all_papers


if __name__ == "__main__":
    download_papers()
