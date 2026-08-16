"""
Vector store layer with two real, fully-implemented backends behind one
interface: Chroma (local, embedded, no server) and PGVector (Postgres +
pgvector extension). Both implement the same ACL-aware retrieval contract:

    query(query_embedding, top_k, allowed_roles) -> list[RetrievedChunk]

ACL enforcement happens at the metadata-filter level of the underlying
store (a real `where` clause in Chroma / a real SQL `WHERE acl_role = ANY(...)`
in Postgres) — not a post-hoc Python filter — so a permission bug in the
store's filter logic will actually show up in retrieval results.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings, get_settings
from app.rag.embeddings import embed_texts


@dataclass
class Document:
    id: str
    text: str
    acl_role: str          # e.g. "Engineering", "HR", "Executive"
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    acl_role: str
    source: str
    score: float            # similarity score (higher = more similar)


class VectorStoreBackend(Protocol):
    def add_documents(self, docs: list[Document]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int, allowed_roles: list[str]) -> list[RetrievedChunk]: ...
    def query_unfiltered(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Used ONLY by the failure-localization/permission-audit layer to detect
        whether ACL filtering blocked or leaked a document — never used to answer
        a real user query."""
        ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class ChromaBackend:
    def __init__(self, settings: Settings | None = None):
        import chromadb

        self.settings = settings or get_settings()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        embeddings = embed_texts([d.text for d in docs], self.settings)
        self.collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=embeddings,
            metadatas=[{"acl_role": d.acl_role, "source": d.source, **d.metadata} for d in docs],
        )

    def _query(self, query_embedding: list[float], top_k: int, where: dict | None) -> list[RetrievedChunk]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res["ids"][0] if res["ids"] else []
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        for i in range(len(ids)):
            # Chroma returns cosine *distance*; convert to a similarity score in [0,1]
            score = 1.0 - float(dists[i])
            out.append(RetrievedChunk(
                id=ids[i], text=docs[i], acl_role=metas[i].get("acl_role", ""),
                source=metas[i].get("source", ""), score=score,
            ))
        return out

    def query(self, query_embedding: list[float], top_k: int, allowed_roles: list[str]) -> list[RetrievedChunk]:
        where = {"acl_role": {"$in": allowed_roles}} if allowed_roles else {"acl_role": "__none__"}
        return self._query(query_embedding, top_k, where)

    def query_unfiltered(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        return self._query(query_embedding, top_k, None)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.settings.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection, metadata={"hnsw:space": "cosine"},
        )


class PGVectorBackend:
    """Postgres + pgvector backend. Requires the `vector` extension to be
    installed on the target database (docker-compose.yml uses the
    `pgvector/pgvector` image which ships it)."""

    def __init__(self, settings: Settings | None = None):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self.settings = settings or get_settings()
        self.conn = psycopg2.connect(self.settings.pg_dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(self.conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        # all-MiniLM-L6-v2 => 384 dims. If you swap embedding_model, update this.
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.settings.pg_table} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    acl_role TEXT NOT NULL,
                    source TEXT,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector(384) NOT NULL
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.settings.pg_table}_embedding_idx
                ON {self.settings.pg_table} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        embeddings = embed_texts([d.text for d in docs], self.settings)
        with self.conn.cursor() as cur:
            for d, emb in zip(docs, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {self.settings.pg_table} (id, text, acl_role, source, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text, acl_role = EXCLUDED.acl_role,
                        source = EXCLUDED.source, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                    """,
                    (d.id, d.text, d.acl_role, d.source, psycopg2_json(d.metadata), emb),
                )

    def _query(self, query_embedding: list[float], top_k: int, role_filter_sql: str, params: tuple) -> list[RetrievedChunk]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, text, acl_role, source, 1 - (embedding <=> %s::vector) AS score
                FROM {self.settings.pg_table}
                {role_filter_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, *params, query_embedding, top_k),
            )
            rows = cur.fetchall()
        return [RetrievedChunk(id=r[0], text=r[1], acl_role=r[2], source=r[3] or "", score=float(r[4])) for r in rows]

    def query(self, query_embedding: list[float], top_k: int, allowed_roles: list[str]) -> list[RetrievedChunk]:
        if not allowed_roles:
            return []
        return self._query(query_embedding, top_k, "WHERE acl_role = ANY(%s)", (allowed_roles,))

    def query_unfiltered(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        return self._query(query_embedding, top_k, "", ())

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.settings.pg_table};")
            return cur.fetchone()[0]

    def reset(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.settings.pg_table};")


def psycopg2_json(d: dict) -> str:
    import json
    return json.dumps(d)


def get_vector_store(settings: Settings | None = None) -> VectorStoreBackend:
    s = settings or get_settings()
    if s.vector_store == "chroma":
        return ChromaBackend(s)
    return PGVectorBackend(s)


def new_id() -> str:
    return str(uuid.uuid4())
