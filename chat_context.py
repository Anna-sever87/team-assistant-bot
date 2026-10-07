"""Контекст чата в оперативной памяти: последние сообщения и сессии «слушания»."""
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger("chat_context")

try:
    from zoneinfo import ZoneInfo

    LOCAL_TZ = ZoneInfo(config.TIMEZONE)
except Exception:  # на Windows без пакета tzdata базы часовых поясов нет
    logger.warning("Часовой пояс %s не найден, использую UTC+7", config.TIMEZONE)
    LOCAL_TZ = timezone(timedelta(hours=7))


@dataclass
class Entry:
    """Одно сообщение чата."""
    message_id: int
    author: str
    user_id: str
    username: str
    ts: int          # время сообщения в секундах (unix)
    text: str


def vector_id(chat_id, message_id) -> str:
    """Идентификатор записи сообщения в Pinecone."""
    return f"{chat_id}:{message_id}"


def format_line(author: str, ts: int, text: str) -> str:
    """Строка вида: [07.10 09:41] Анна: текст."""
    moment = datetime.fromtimestamp(int(ts), LOCAL_TZ).strftime("%d.%m %H:%M")
    return f"[{moment}] {author}: {text}"


def format_entries(entries: List[Entry]) -> str:
    return "\n".join(format_line(e.author, e.ts, e.text) for e in entries)


class ChatContext:
    def __init__(self, recent_limit: int = config.RECENT_MESSAGES) -> None:
        self._recent: Dict[str, deque] = defaultdict(lambda: deque(maxlen=recent_limit))
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def add(self, chat_id, entry: Entry) -> Optional[str]:
        """Запоминает сообщение. Возвращает id сессии слушания, если она идёт."""
        key = str(chat_id)
        with self._lock:
            self._recent[key].append(entry)
            session = self._sessions.get(key)
            if session:
                session["entries"].append(entry)
                return session["id"]
        return None

    def start_listening(self, chat_id) -> Tuple[str, bool]:
        """Включает слушание. Возвращает (id сессии, уже ли слушали до этого)."""
        key = str(chat_id)
        with self._lock:
            if key in self._sessions:
                return self._sessions[key]["id"], True
            session_id = datetime.now(timezone.utc).strftime("s%Y%m%d%H%M%S")
            self._sessions[key] = {"id": session_id, "entries": []}
            return session_id, False

    def stop_listening(self, chat_id) -> Optional[List[Entry]]:
        """Выключает слушание. Возвращает сообщения сессии или None, если слушание не шло."""
        with self._lock:
            session = self._sessions.pop(str(chat_id), None)
        return session["entries"] if session else None

    def is_listening(self, chat_id) -> bool:
        with self._lock:
            return str(chat_id) in self._sessions

    def recent(self, chat_id, exclude_message_id: Optional[int] = None) -> List[Entry]:
        with self._lock:
            entries = list(self._recent[str(chat_id)])
        return [e for e in entries if e.message_id != exclude_message_id]