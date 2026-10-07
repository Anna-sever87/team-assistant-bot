"""PipelineManager: три пайплайна Haystack (индексация, поиск, резюмирование)."""
import logging
import threading
from typing import List, Optional

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

import config
from chat_context import Entry, format_entries, vector_id
from components import PineconeRetriever, PineconeWriter, PromptBuilder
from vector_store import TYPE_CHAT_MESSAGE, PineconeStore

logger = logging.getLogger("pipelines")


class PipelineManager:
    def __init__(self, store: PineconeStore) -> None:
        self.store = store
        self.indexing_pipeline = self._create_indexing_pipeline()
        self.query_pipeline = self._create_query_pipeline()
        self.summarization_pipeline = self._create_summarization_pipeline()
        # Пайплайн нельзя запускать из нескольких потоков одновременно.
        self._locks = {name: threading.Lock() for name in ("indexing", "query", "summarization")}

    # ---------- общие настройки компонентов (для каждого пайплайна создаются НОВЫЕ объекты) ----------
    @staticmethod
    def _api_key() -> Secret:
        return Secret.from_token(config.OPENAI_API_KEY)

    def _text_embedder(self) -> OpenAITextEmbedder:
        return OpenAITextEmbedder(
            api_key=self._api_key(), model=config.EMBEDDING_MODEL, api_base_url=config.OPENAI_BASE_URL
        )

    # ---------- 1. Пайплайн индексации: embedder -> writer ----------
    def _create_indexing_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("embedder", OpenAIDocumentEmbedder(
            api_key=self._api_key(), model=config.EMBEDDING_MODEL, api_base_url=config.OPENAI_BASE_URL
        ))
        pipeline.add_component("writer", PineconeWriter(self.store))
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    # ---------- 2. Пайплайн поиска: text_embedder -> retriever ----------
    def _create_query_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self._text_embedder())
        pipeline.add_component("retriever", PineconeRetriever(self.store, config.FIND_TOP_K, config.MIN_SCORE))
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        return pipeline

    # ---------- 3. Пайплайн резюмирования: text_embedder -> retriever -> prompt_builder -> llm ----------
    def _create_summarization_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self._text_embedder())
        pipeline.add_component("retriever", PineconeRetriever(self.store, config.SEARCH_TOP_K, config.MIN_SCORE))
        pipeline.add_component("prompt_builder", PromptBuilder())
        pipeline.add_component("llm", OpenAIChatGenerator(
            api_key=self._api_key(),
            model=config.OPENAI_MODEL,
            api_base_url=config.OPENAI_BASE_URL,
            generation_kwargs={"temperature": 0.5, "max_tokens": 900},
        ))
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.messages", "llm.messages")
        return pipeline

    # ---------- методы для бота ----------
    def index_entry(self, chat_id, entry: Entry, session_id: Optional[str] = None) -> None:
        """Сохраняет сообщение чата в Pinecone вместе с метаданными автора."""
        document = Document(
            content=f"{entry.author}: {entry.text}",   # автор попадает в эмбеддинг: можно искать «что говорила Анна»
            meta={
                "vector_id": vector_id(chat_id, entry.message_id),
                "type": TYPE_CHAT_MESSAGE,
                "chat_id": str(chat_id),
                "message_id": entry.message_id,
                "author": entry.author,
                "username": entry.username,
                "user_id": entry.user_id,
                "ts": entry.ts,
                "text": entry.text,
                "session_id": session_id,
            },
        )
        with self._locks["indexing"]:
            self.indexing_pipeline.run({"embedder": {"documents": [document]}})

    def find(self, chat_id, text: str) -> List[Document]:
        """Ищет в истории чата сообщения по смыслу запроса."""
        with self._locks["query"]:
            result = self.query_pipeline.run({
                "text_embedder": {"text": text[: config.EMBED_QUERY_CHARS]},
                "retriever": {"chat_id": str(chat_id)},
            })
        return result["retriever"]["documents"]

    def summarize(
        self,
        chat_id,
        mode: str,
        dialogue: List[Entry],
        query_text: str,
        question: str = "",
        asker: str = "",
    ) -> str:
        """mode='summary': итог обсуждения; mode='mention': ответ на обращение к боту."""
        dialogue_text = format_entries(dialogue)
        if len(dialogue_text) > config.MAX_DIALOGUE_CHARS:
            dialogue_text = "(начало диалога обрезано)\n" + dialogue_text[-config.MAX_DIALOGUE_CHARS:]
        exclude_ids = [vector_id(chat_id, e.message_id) for e in dialogue]
        with self._locks["summarization"]:
            result = self.summarization_pipeline.run({
                "text_embedder": {"text": query_text[: config.EMBED_QUERY_CHARS] or question or "обсуждение"},
                "retriever": {"chat_id": str(chat_id), "exclude_ids": exclude_ids},
                "prompt_builder": {
                    "mode": mode, "dialogue": dialogue_text, "question": question, "asker": asker,
                },
            })
        reply = (result["llm"]["replies"][0].text or "").strip()
        return reply or "Не получилось сформулировать ответ."