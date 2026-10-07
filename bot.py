"""Командный Telegram-бот: слушает чат, подводит итоги обсуждений и отвечает на обращения.

Запуск: python bot.py
"""
import logging
import re
import threading
from typing import List, Optional

import telebot

import config
from chat_context import ChatContext, Entry, format_line
from pipelines import PipelineManager
from vector_store import PineconeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("bot")

config.check_settings()

HELP_TEXT = (
    "Я помощник рабочей команды. Что я умею:\n"
    "- запоминаю все сообщения чата вместе с авторами;\n"
    "- /start_listening: начать слушать обсуждение;\n"
    "- /stop_listening: закончить и прислать итог (тему, тезисы, мнение по спору или принятое решение);\n"
    "- /find текст: найти в истории чата сообщения по смыслу;\n"
    "- если обратиться ко мне через @ или ответить на моё сообщение, я отвечу по контексту переписки;\n"
    "- /chat_id: показать ID этого чата."
)

bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
store = PineconeStore()
manager = PipelineManager(store)
context = ChatContext()
me = bot.get_me()
BOT_USERNAME = (me.username or "").lower()


# ---------- вспомогательные функции ----------
def is_allowed(message) -> bool:
    """Личные сообщения: только пользователи из ALLOWED_USER_IDS. Группы: чаты из ALLOWED_CHAT_IDS (пусто = все)."""
    if message.chat.type == "private":
        user_id = message.from_user.id if message.from_user else None
        if user_id in config.ALLOWED_USER_IDS:
            return True
        logger.warning("Личное сообщение от неразрешённого пользователя id=%s. Игнорирую.", user_id)
        return False
    if not config.ALLOWED_CHAT_IDS or message.chat.id in config.ALLOWED_CHAT_IDS:
        return True
    logger.warning("Сообщение из неразрешённого чата id=%s. Игнорирую.", message.chat.id)
    return False


def author_name(user) -> str:
    if user is None:
        return "Неизвестный"
    full_name = " ".join(part for part in (user.first_name, user.last_name) if part).strip()
    return full_name or user.username or str(user.id)


def make_entry(message) -> Entry:
    user = message.from_user
    return Entry(
        message_id=message.message_id,
        author=author_name(user),
        user_id=str(user.id) if user else "",
        username=(user.username or "") if user else "",
        ts=int(message.date),
        text=(message.text or "").strip(),
    )


def is_for_bot(message) -> bool:
    """Обращение к боту: личный чат, упоминание через @ или ответ на сообщение бота."""
    if message.chat.type == "private":
        return True
    if BOT_USERNAME and f"@{BOT_USERNAME}" in (message.text or "").lower():
        return True
    reply = message.reply_to_message
    return bool(reply and reply.from_user and reply.from_user.id == me.id)


def strip_mention(text: str) -> str:
    cleaned = re.sub(rf"@{re.escape(BOT_USERNAME)}", "", text, flags=re.IGNORECASE).strip() if BOT_USERNAME else text
    return cleaned or "Что думаешь?"


def send_long(chat_id, text: str, reply_to: Optional[int] = None) -> None:
    """Отправляет длинный текст частями (лимит Telegram 4096 символов)."""
    parts: List[str] = []
    while text:
        if len(text) <= config.TELEGRAM_LIMIT:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, config.TELEGRAM_LIMIT)
        cut = cut if cut > 0 else config.TELEGRAM_LIMIT
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    for index, part in enumerate(parts):
        bot.send_message(chat_id, part, reply_to_message_id=reply_to if index == 0 else None)


def index_in_background(chat_id, entry: Entry, session_id: Optional[str]) -> None:
    """Запись в Pinecone идёт в отдельном потоке, чтобы не задерживать ответы бота."""
    def job() -> None:
        try:
            manager.index_entry(chat_id, entry, session_id)
        except Exception:
            logger.exception("Не удалось сохранить сообщение в Pinecone")

    threading.Thread(target=job, daemon=True).start()


# ---------- команды ----------
@bot.message_handler(commands=["chat_id"])
def handle_chat_id(message):
    bot.reply_to(message, f"ID этого чата: {message.chat.id}\nТип: {message.chat.type}\nТвой ID: {message.from_user.id}")


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    if is_allowed(message):
        bot.send_message(message.chat.id, HELP_TEXT)


@bot.message_handler(commands=["start_listening"])
def handle_start_listening(message):
    if not is_allowed(message):
        return
    _, already = context.start_listening(message.chat.id)
    if already:
        bot.reply_to(message, "Я уже слушаю это обсуждение. Когда закончите, напишите /stop_listening.")
    else:
        bot.reply_to(message, "Слушаю. Обсуждайте, а когда закончите, напишите /stop_listening, и я подведу итог.")
    logger.info("Слушание включено, чат %s", message.chat.id)


@bot.message_handler(commands=["stop_listening"])
def handle_stop_listening(message):
    if not is_allowed(message):
        return
    entries = context.stop_listening(message.chat.id)
    if entries is None:
        bot.reply_to(message, "Я сейчас не слушаю. Сначала напишите /start_listening.")
        return
    if not entries:
        bot.reply_to(message, "Пока никто ничего не написал, итог подводить не из чего.")
        return
    try:
        bot.send_message(message.chat.id, f"Слушание остановлено. Анализирую сообщений: {len(entries)}…")
        bot.send_chat_action(message.chat.id, "typing")
        summary = manager.summarize(
            message.chat.id, "summary", entries, query_text="\n".join(e.text for e in entries)
        )
        send_long(message.chat.id, summary)
    except Exception:
        logger.exception("Ошибка при подведении итога")
        bot.send_message(message.chat.id, "Не получилось подвести итог. Попробуйте ещё раз чуть позже.")


@bot.message_handler(commands=["find"])
def handle_find(message):
    if not is_allowed(message):
        return
    query = (message.text or "").partition(" ")[2].strip()
    if not query:
        bot.reply_to(message, "Напишите, что искать: /find текст")
        return
    try:
        documents = manager.find(message.chat.id, query)
        if not documents:
            bot.reply_to(message, "Ничего похожего в истории чата не нашёл.")
            return
        lines = [format_line(d.meta.get("author", "?"), d.meta.get("ts", 0), d.content) for d in documents]
        send_long(message.chat.id, "Нашёл в истории чата:\n" + "\n".join(lines), reply_to=message.message_id)
    except Exception:
        logger.exception("Ошибка поиска")
        bot.reply_to(message, "Не получилось выполнить поиск. Попробуйте ещё раз чуть позже.")


# ---------- обычные сообщения ----------
@bot.message_handler(func=lambda m: m.content_type == "text" and not (m.text or "").startswith("/"))
def handle_text(message):
    if not is_allowed(message) or not (message.text or "").strip():
        return
    if message.from_user and message.from_user.is_bot:
        return

    entry = make_entry(message)
    dialogue = context.recent(message.chat.id, exclude_message_id=entry.message_id)
    session_id = context.add(message.chat.id, entry)
    index_in_background(message.chat.id, entry, session_id)

    if not is_for_bot(message):
        return
    question = strip_mention(entry.text)
    try:
        bot.send_chat_action(message.chat.id, "typing")
        query_text = "\n".join(e.text for e in dialogue[-5:] + [entry]) if dialogue else question
        answer = manager.summarize(
            message.chat.id, "mention", dialogue, query_text=query_text, question=question, asker=entry.author
        )
        send_long(message.chat.id, answer, reply_to=message.message_id)
    except Exception:
        logger.exception("Ошибка при ответе на обращение")
        bot.reply_to(message, "Что-то пошло не так. Попробуйте ещё раз чуть позже.")


def setup_commands() -> None:
    try:
        bot.set_my_commands([
            telebot.types.BotCommand("start_listening", "Начать слушать обсуждение"),
            telebot.types.BotCommand("stop_listening", "Закончить и получить итог"),
            telebot.types.BotCommand("find", "Найти сообщения по смыслу"),
            telebot.types.BotCommand("chat_id", "Показать ID чата"),
            telebot.types.BotCommand("help", "Что умеет бот"),
        ])
    except Exception:
        logger.warning("Не удалось обновить список команд бота", exc_info=True)


if __name__ == "__main__":
    setup_commands()
    if getattr(me, "can_read_all_group_messages", None) is False:
        logger.warning(
            "Бот не видит все сообщения групп. В @BotFather отключите privacy mode: "
            "/setprivacy -> выберите бота -> Disable, затем удалите бота из группы и добавьте снова."
        )
    logger.info("Бот @%s запущен. Индекс: %s, пространство: %s",
                me.username, store.index_name, store.namespace)
    bot.infinity_polling(skip_pending=True)