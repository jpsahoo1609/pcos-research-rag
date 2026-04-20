-- Run this in Supabase SQL Editor BEFORE running ingest.py

-- Enable vector extension
create extension if not exists vector;

-- Create documents table
create table if not exists documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(1536)
);

-- Create search function
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb DEFAULT '{}'
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create index for fast search
create index if not exists documents_embedding_idx
on documents using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
