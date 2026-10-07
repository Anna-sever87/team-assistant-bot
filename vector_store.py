"""PineconeStore: подключение к индексу Pinecone, запись и поиск сообщений чата."""
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from pinecone import Pinecone, ServerlessSpec

import config

logger = logging.getLogger("vector_store")

TYPE_CHAT_MESSAGE = "chat_message"


def clean_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Убирает пустые значения: Pinecone не принимает None."""
    cleaned = {}
    for key, value in (metadata or {}).items():
        if value is None or value == "":
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


class PineconeStore:
    def __init__(self) -> None:
        self.index_name = config.PINECONE_INDEX_NAME
        self.namespace = config.PINECONE_NAMESPACE
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._connect_index()

    def _connect_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            logger.info("Создаю индекс %s ...", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        return self.pc.Index(self.index_name)

    def upsert(self, items: List[Dict[str, Any]]) -> int:
        """Записывает вектора пачками. items: [{id, values, metadata}]."""
        total = 0
        for start in range(0, len(items), 100):
            batch = items[start:start + 100]
            self.index.upsert(
                vectors=[
                    {"id": item["id"], "values": item["values"], "metadata": clean_metadata(item["metadata"])}
                    for item in batch
                ],
                namespace=self.namespace,
            )
            total += len(batch)
        return total

    def query(
        self, vector: List[float], chat_id: str, top_k: int, exclude_ids: Iterable[str] = ()
    ) -> List[Dict[str, Any]]:
        """Ищет похожие сообщения этого чата. exclude_ids: записи, которые нужно пропустить."""
        exclude = set(exclude_ids)
        result = self.index.query(
            vector=vector,
            top_k=min(top_k + len(exclude), 200),
            include_metadata=True,
            namespace=self.namespace,
            filter={"chat_id": {"$eq": str(chat_id)}, "type": {"$eq": TYPE_CHAT_MESSAGE}},
        )
        matches = [
            {"id": m.id, "score": round(m.score, 4), "metadata": dict(m.metadata or {})}
            for m in result.matches
            if m.id not in exclude
        ]
        return matches[:top_k]

    def stats(self) -> Any:
        return self.index.describe_index_stats()