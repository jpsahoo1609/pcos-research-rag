import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(filepath="data/papers.json"):
    with open(filepath) as f:
        papers = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for paper in papers:
        splits = splitter.split_text(paper["text"])
        for split in splits:
            chunks.append({
                "content": split,
                "metadata": {
                    "pmcid": paper["pmcid"],
                    "section": paper["section"],
                    "source": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['pmcid']}/"
                }
            })

    print(f"Created {len(chunks)} chunks from {len(papers)} passages")
    return chunks
