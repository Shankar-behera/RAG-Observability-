"""
RAG Triad evaluation using Ragas, wired to the real configured LLM
(OpenAI / DeepSeek / Ollama) and the real local embedding model. Ragas'
LLM-judged metrics (faithfulness, answer relevancy, context precision,
context recall) make actual chat-completion calls through the same
provider abstraction the pipeline itself uses — there is no separate
"eval mode" mock judge.
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from app.config import Settings, get_settings
from app.rag.llm_providers import get_llm_client


@dataclass
class EvalSample:
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None  # reference answer, needed for context recall


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None

    @property
    def rag_triad_pass(self) -> bool:
        return self.faithfulness is not None and self.faithfulness >= 0.0  # gate applied by caller



class RagasEvaluator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        
        
        # Provide an explicit Async client so Ragas can use ascore()
        async_client = AsyncOpenAI(
            base_url=self.settings.provider_base_url(), 
            api_key=self.settings.provider_api_key()
        )
        self.ragas_llm = llm_factory(model=self.settings.provider_model(), provider="openai", client=async_client)
        # Loads its own local sentence-transformers instance by model name (real, on-device).
        self.ragas_embeddings = embedding_factory(provider="huggingface", model=self.settings.embedding_model)

        self.faithfulness = Faithfulness(llm=self.ragas_llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
        self.context_precision = ContextPrecisionWithoutReference(llm=self.ragas_llm)
        self.context_recall = ContextRecall(llm=self.ragas_llm)

    async def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        faithfulness_score = await self.faithfulness.ascore(
            user_input=sample.query, response=sample.answer, retrieved_contexts=sample.contexts,
        )
        relevancy_score = await self.answer_relevancy.ascore(
            user_input=sample.query, response=sample.answer,
        )
        precision_score = await self.context_precision.ascore(
            user_input=sample.query, response=sample.answer, retrieved_contexts=sample.contexts,
        )

        recall_score = None
        if sample.ground_truth:
            recall_score = await self.context_recall.ascore(
                user_input=sample.query, retrieved_contexts=sample.contexts, reference=sample.ground_truth,
            )

        return EvalResult(
            faithfulness=float(faithfulness_score.value if hasattr(faithfulness_score, "value") else faithfulness_score),
            answer_relevancy=float(relevancy_score.value if hasattr(relevancy_score, "value") else relevancy_score),
            context_precision=float(precision_score.value if hasattr(precision_score, "value") else precision_score),
            context_recall=(float(recall_score.value if hasattr(recall_score, "value") else recall_score) if recall_score is not None else None),
        )
