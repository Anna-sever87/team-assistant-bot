"""Свои компоненты Haystack: запись и поиск в Pinecone, сборка промпта."""
import logging
from typing import Dict, List, Optional

from haystack import Document, component
from haystack.dataclasses import ChatMessage

import prompts
from chat_context import format_line
from vector_store import PineconeStore, clean_metadata

logger = logging.getLogger("components")


@component
class PineconeWriter:
    """Записывает документы с эмбеддингами в Pinecone."""

    def __init__(self, store: PineconeStore) -> None:
        self.store = store

    @component.output_types(written=int)
    def run(self, documents: List[Document]) -> Dict[str, int]:
        items = []
        for doc in documents:
            if doc.embedding is None:
                continue
            meta = dict(doc.meta)
            items.append({
                "id": meta.pop("vector_id"),
                "values": list(doc.embedding),
                "metadata": clean_metadata(meta),
            })
        written = self.store.upsert(items)
        logger.info("В Pinecone записано сообщений: %s", written)
        return {"written": written}


@component
class PineconeRetriever:
    """Ищет в Pinecone сообщения конкретного чата по вектору запроса."""

    def __init__(self, store: PineconeStore, top_k: int, min_score: float) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    @component.output_types(documents=List[Document])
    def run(
        self, query_embedding: List[float], chat_id: str, exclude_ids: Optional[List[str]] = None
    ) -> Dict[str, List[Document]]:
        matches = self.store.query(query_embedding, chat_id, self.top_k, exclude_ids or [])
        documents = []
        for match in matches:
            meta = match["metadata"]
            if match["score"] < self.min_score or not meta.get("text"):
                continue
            documents.append(Document(content=meta["text"], meta=meta, score=match["score"]))
        logger.info("Поиск: найдено %s из %s", len(documents), len(matches))
        return {"documents": documents}


def format_found(documents: List[Document]) -> str:
    """Найденные сообщения в хронологическом порядке, по одному в строке."""
    ordered = sorted(documents, key=lambda d: d.meta.get("ts", 0))
    return "\n".join(
        format_line(d.meta.get("author", "?"), d.meta.get("ts", 0), d.content) for d in ordered
    )


@component
class PromptBuilder:
    """Собирает промпт для модели: режим 'summary' (итог обсуждения) или 'mention' (ответ на обращение)."""

    @component.output_types(messages=List[ChatMessage])
    def run(
        self,
        mode: str,
        dialogue: str,
        documents: List[Document],
        question: str = "",
        asker: str = "",
    ) -> Dict[str, List[ChatMessage]]:
        template = prompts.SUMMARY_TEMPLATE if mode == "summary" else prompts.MENTION_TEMPLATE
        text = template.format(
            dialogue=dialogue.strip() or prompts.EMPTY_DIALOGUE,
            history=format_found(documents) or prompts.EMPTY_HISTORY,
            question=question,
            asker=asker or "участник",
        )
        return {"messages": [ChatMessage.from_system(prompts.SYSTEM_PROMPT), ChatMessage.from_user(text)]}