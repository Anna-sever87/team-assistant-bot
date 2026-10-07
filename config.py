"""Настройки проекта: ключи и модели из .env, остальные параметры здесь."""
import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _ids(name: str) -> set:
    """Читает из .env список числовых ID через запятую (ID групп бывают отрицательными)."""
    items = os.getenv(name, "").replace(" ", "").split(",")
    return {int(item) for item in items if item.lstrip("-").isdigit()}


# ---------- ключи и модели (из .env) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_DIM = 1536   # размерность text-embedding-3-small; при смене модели поменяйте и индекс

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "haystack-memory"
PINECONE_NAMESPACE = "team-chat"   # отдельное пространство имён внутри индекса

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Кто может пользоваться ботом.
# В личной переписке: только пользователи из ALLOWED_USER_IDS.
# В группах: только чаты из ALLOWED_CHAT_IDS (если список пустой, разрешены все группы).
ALLOWED_USER_IDS = _ids("ALLOWED_USER_IDS")
ALLOWED_CHAT_IDS = _ids("ALLOWED_CHAT_IDS")

# ---------- поведение бота ----------
TIMEZONE = os.getenv("TIMEZONE") or "Asia/Novosibirsk"   # часовой пояс для времени сообщений
RECENT_MESSAGES = 30       # сколько последних сообщений чата помнить в оперативной памяти
SEARCH_TOP_K = 6           # сколько найденных ранее сообщений добавлять в промпт
FIND_TOP_K = 5             # сколько результатов выдавать по команде /find
MIN_SCORE = 0.2            # слишком слабые совпадения не берём
MAX_DIALOGUE_CHARS = 12000   # больше этого диалог обрезается (берём конец)
EMBED_QUERY_CHARS = 6000     # сколько текста отдаём на эмбеддинг поискового запроса
TELEGRAM_LIMIT = 4000        # максимум символов в одном сообщении Telegram (лимит 4096)


def check_settings() -> None:
    """Останавливает запуск, если в .env чего-то не хватает."""
    missing = [
        name for name, value in (
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
        ) if not value
    ]
    if missing:
        raise SystemExit("В файле .env не хватает: " + ", ".join(missing))