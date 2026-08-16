from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import IngestRequest, IngestResponse
from app.rag.vector_store import Document, get_vector_store, new_id

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest_documents(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    store = get_vector_store(settings)
    docs = [
        Document(id=new_id(), text=d.text, acl_role=d.acl_role, source=d.source, metadata=d.metadata)
        for d in req.documents
    ]
    store.add_documents(docs)
    return IngestResponse(ingested=len(docs), total_in_store=store.count())
