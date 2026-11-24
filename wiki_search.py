"""
wiki_search.py — поиск подтверждений в Википедии.

Это НЕ агент и НЕ нейросеть, а обычный код. Он делает три вещи:
  1. Ищет в Википедии статьи по тексту факта (как поиск на сайте Википедии).
  2. Скачивает текст найденных статей.
  3. Выбирает из статей несколько абзацев, по которым можно проверить факт.

Зачем выбирать абзацы, а не отдавать модели всю статью целиком:
статья может быть огромной, а каждый символ, отправленный модели, тратит
дневной лимит токенов Groq. Поэтому отдаём только самое нужное.

Как выбираются абзацы:
  - ПЕРВЫЙ абзац каждой статьи (вступление) получает бонус — там обычно
    собраны главные факты: даты, место рождения, чем известен;
  - остальные абзацы ранжируются по сходству с фактом. Способ задаётся
    в config.py (WIKI_RANKING):
      "overlap" — число совпавших слов (первая, простая версия);
      "bm25"    — алгоритм BM25: редкие слова важнее частых (улучшенная версия).
В полной версии этот шаг можно улучшить ещё (например, искать по смыслу
с помощью эмбеддингов) — это один из «рычагов» повышения качества.

Бережное отношение к Википедии:
  - между запросами делается небольшая пауза;
  - если Википедия ответила «слишком много запросов» (ошибка 429),
    программа ждёт и повторяет попытку;
  - все ответы сохраняются на диск (папка cache/wiki), поэтому повторный
    поиск того же самого не обращается к Википедии вообще. Это сильно
    ускоряет эксперименты и делает их воспроизводимыми.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import requests

import config

# Википедия просит, чтобы программы представлялись и указывали контакт.
# Программам без контакта Википедия чаще отвечает «подождите» (ошибка 429).
_contact = f"; {config.WIKI_CONTACT}" if config.WIKI_CONTACT else ""
HEADERS = {"User-Agent": f"HallucinationDemo/0.3 (master's thesis student project{_contact}) python-requests"}

CACHE_DIR = Path("cache") / "wiki"

# Частые слова, которые не несут смысла для поиска совпадений
STOPWORDS = {
    # английские
    "the", "and", "was", "were", "is", "are", "for", "with", "that", "this",
    "from", "has", "had", "have", "his", "her", "its", "their", "which", "who",
    "been", "also", "into", "than", "then", "there", "they", "she", "him", "not",
    # русские
    "это", "был", "была", "было", "были", "который", "которая", "которые",
    "его", "её", "она", "они", "оно", "что", "как", "для", "при", "также",
    "так", "или", "над", "под", "без", "после", "году", "года",
}

LEAD_BONUS = 3          # бонус вступлению статьи при простом подсчёте слов
LEAD_BONUS_SHARE = 0.5  # бонус вступлению при BM25: половина от лучшей оценки

_session = requests.Session()
_session.headers.update(HEADERS)
_last_request_time = 0.0


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Если в тексте есть русские буквы — ищем в русской Википедии, иначе в английской."""
    return "ru" if re.search(r"[а-яА-ЯёЁ]", text) else "en"


def _keywords(text: str) -> set[str]:
    """
    Превращает текст в набор «ключевых слов».
    Берём только первые 5 букв каждого слова — грубый, но рабочий способ
    не различать формы слова: «родился», «родилась» → «родил».
    """
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    return {w[:5] for w in words if len(w) > 2 and w not in STOPWORDS}


def _wiki_request(lang: str, params: dict) -> dict:
    """
    Один запрос к Википедии с кэшем на диске, паузами и повторами.
    Все остальные функции обращаются к Википедии только через неё.
    """
    global _last_request_time

    # 1. Смотрим, нет ли ответа в кэше
    key = hashlib.sha1(json.dumps([lang, params], sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"https://{lang}.wikipedia.org/w/api.php"

    for attempt in range(5):
        # 2. Пауза между запросами, чтобы не перегружать Википедию
        wait = config.WIKI_REQUEST_DELAY - (time.time() - _last_request_time)
        if wait > 0:
            time.sleep(wait)

        response = _session.get(url, params=params, timeout=20)
        _last_request_time = time.time()

        # 3. Если Википедия просит притормозить — ждём и пробуем снова
        if response.status_code == 429 or response.status_code >= 500:
            delay = int(response.headers.get("Retry-After", 0)) or 2 ** (attempt + 1)
            print(f"[wiki] Википедия просит подождать, пауза {delay} с...")
            time.sleep(delay)
            continue

        response.raise_for_status()
        data = response.json()

        # 4. Сохраняем ответ в кэш
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    raise requests.RequestException("Википедия не отвечает после 5 попыток")


# ---------------------------------------------------------------------------
# Работа с Википедией
# ---------------------------------------------------------------------------

def search_titles(query: str, lang: str = "en", limit: int = 3) -> list[str]:
    """Найти названия статей по запросу (как строка поиска на сайте Википедии)."""
    data = _wiki_request(lang, {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    })
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def get_article_text(title: str, lang: str = "en") -> str:
    """Скачать текст статьи (без картинок и разметки)."""
    data = _wiki_request(lang, {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,   # обычный текст, без HTML
        "redirects": 1,     # если статья — перенаправление, перейти к настоящей
        "titles": title,
        "format": "json",
    })
    for page in data.get("query", {}).get("pages", {}).values():
        return page.get("extract", "") or ""
    return ""


def _split_paragraphs(text: str, max_len: int = 700) -> list[str]:
    """Разбить статью на небольшие фрагменты БЕЗ потери конца длинных абзацев.

    В старой версии использовалось ``line[:700]``. Поэтому нужная фраза,
    находящаяся после 700-го символа, вообще не доходила до верификатора.
    Здесь длинный абзац делится на отдельные предложения/фрагменты.
    """
    paragraphs: list[str] = []

    def append_long_segment(segment: str) -> None:
        """Разбить очень длинное предложение по границам слов."""
        remaining = segment.strip()
        while len(remaining) > max_len:
            cut = remaining.rfind(" ", 0, max_len + 1)
            if cut < max_len // 2:
                cut = max_len
            fragment = remaining[:cut].strip()
            if fragment:
                paragraphs.append(fragment)
            remaining = remaining[cut:].strip()
        if remaining:
            paragraphs.append(remaining)

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        # Короткие (но содержательные) предложения тоже могут быть доказательствами.
        if len(line) < 25 or line.startswith("=="):
            continue
        if len(line) <= max_len:
            paragraphs.append(line)
            continue

        # Стараемся не разрывать предложения посередине.
        sentences = re.split(r"(?<=[.!?。！？])\s+", line)
        chunk = ""
        for sentence in sentences:
            if len(sentence) > max_len:
                if chunk:
                    paragraphs.append(chunk)
                    chunk = ""
                append_long_segment(sentence)
            elif not chunk:
                chunk = sentence
            elif len(chunk) + 1 + len(sentence) <= max_len:
                chunk += " " + sentence
            else:
                paragraphs.append(chunk)
                chunk = sentence
        if chunk:
            paragraphs.append(chunk)
    return paragraphs


# ---------------------------------------------------------------------------
# Главная функция, которой будет пользоваться агент-верификатор
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """Как _keywords, но списком с повторами (нужно для подсчёта частоты слов)."""
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    return [w[:5] for w in words if len(w) > 2 and w not in STOPWORDS]


def _bm25_scores(claim: str, paragraphs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Оценка абзацев методом BM25 — классический алгоритм поисковых систем.

    Чем он лучше простого подсчёта совпавших слов:
      1. Редкие слова важнее частых. Если «Bell» встречается почти в каждом
         абзаце статьи про Белла, совпадение по нему мало что значит.
         А слово «died» встречается редко — абзац, где оно есть,
         скорее всего, и нужен для проверки факта о смерти.
      2. Длинные абзацы не получают преимущества просто за длину.
    """
    query_terms = set(_tokens(claim))
    docs = [_tokens(p) for p in paragraphs]
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n if n else 1

    # В скольких абзацах встречается каждое слово факта
    doc_freq = {t: sum(1 for d in docs if t in d) for t in query_terms}
    # «Вес» слова: чем реже встречается, тем больше
    idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1) for t, df in doc_freq.items()}

    scores = []
    for doc in docs:
        score = 0.0
        for term in query_terms:
            tf = doc.count(term)
            if tf:
                score += idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


def _search_one(query: str, lang: str) -> list[str]:
    """Поиск по одному запросу; если ничего не нашлось — повтор только по именам и названиям."""
    titles = search_titles(query, lang, config.WIKI_ARTICLES_PER_QUERY)
    if not titles:
        # Слова с заглавной буквы — обычно имена и названия
        names = " ".join(re.findall(r"\b[A-ZА-ЯЁ][\w-]+", query))
        if names and names != query:
            titles = search_titles(names, lang, config.WIKI_ARTICLES_PER_QUERY)
    return titles


def find_evidence(
    claim: str,
    query: str | list[str] | None = None,
    lang: str | None = None,
) -> dict:
    """
    Найти в Википедии текст, по которому можно проверить факт.

    claim — сам факт («Белл родился в Лондоне»);
    query — что искать: один запрос или список запросов
            (их формулирует агент-экстрактор, например
            ["Alexander Graham Bell", "AT&T"]). Если не задано —
            ищем по тексту факта;
    lang  — "ru" или "en" (если не задано, определяется автоматически).

    Режим поиска задаётся в config.py (WIKI_MULTI_QUERY):
      False — ищем только по первому запросу (простой режим);
      True  — ищем по всем запросам И по тексту факта, абзацы из всех
              найденных статей собираются вместе (улучшенный режим).

    Возвращает словарь:
      {
        "text":    выбранные абзацы, склеенные в один текст,
        "sources": [{"title": ..., "url": ...}, ...],
      }
    Если ничего не найдено, "text" будет пустой строкой.
    """
    lang = lang or detect_language(claim)
    claim_words = _keywords(claim)

    # Приводим запрос(ы) к списку
    if query is None:
        queries = [claim]
    elif isinstance(query, str):
        queries = [query]
    else:
        queries = [q for q in query if q] or [claim]

    if config.WIKI_MULTI_QUERY:
        if claim not in queries:
            queries = queries + [claim]   # дополнительно ищем по тексту факта
    else:
        queries = queries[:1]             # простой режим: только первый запрос

    # Собираем названия статей из разных запросов по очереди.
    # Раньше первые запросы могли занять ВСЕ доступные места, и результаты
    # остальных запросов отбрасывались из-за WIKI_MAX_ARTICLES.
    results_by_query: list[list[str]] = []
    for q in queries:
        try:
            results_by_query.append(_search_one(q, lang))
        except requests.RequestException as error:
            print(f"[wiki] Ошибка поиска по запросу «{q}»: {error}")
            results_by_query.append([])

    titles: list[str] = []
    # Если есть три поисковых запроса, берём сначала по одному результату
    # каждого запроса, затем вторые результаты и т. д.
    max_results = max((len(items) for items in results_by_query), default=0)
    for rank in range(max_results):
        for items in results_by_query:
            if rank < len(items) and items[rank] not in titles:
                titles.append(items[rank])
                if len(titles) >= config.WIKI_MAX_ARTICLES:
                    break
        if len(titles) >= config.WIKI_MAX_ARTICLES:
            break

    # Собираем абзацы из всех найденных статей
    paragraphs = []  # (название статьи, абзац, это вступление?)
    for title in titles:
        try:
            article = get_article_text(title, lang)
        except requests.RequestException as error:
            print(f"[wiki] Не удалось загрузить статью «{title}»: {error}")
            continue
        for index, paragraph in enumerate(_split_paragraphs(article)):
            paragraphs.append((title, paragraph, index == 0))

    if not paragraphs:
        return {"text": "", "sources": []}

    # Оцениваем, насколько каждый абзац подходит для проверки факта
    if config.WIKI_RANKING == "bm25":
        scores = _bm25_scores(claim, [p for _, p, _ in paragraphs])
        lead_bonus = LEAD_BONUS_SHARE * max(scores) if scores else 0
    else:  # "overlap" — простой подсчёт совпавших слов (первая версия)
        scores = [len(claim_words & _keywords(p)) for _, p, _ in paragraphs]
        lead_bonus = LEAD_BONUS

    candidates = [
        (score + (lead_bonus if is_lead else 0), title, paragraph)
        for score, (title, paragraph, is_lead) in zip(scores, paragraphs)
    ]

    # Самые полезные абзацы — наверх
    candidates.sort(key=lambda item: item[0], reverse=True)

    # Выбираем подходящие абзацы, соблюдая лимит текста.
    # Прежде использовался break при первом слишком длинном абзаце:
    # тогда более короткие релевантные абзацы ниже по списку терялись.
    chosen, used_titles, total_len = [], [], 0
    seen_passages: set[str] = set()
    for score, title, paragraph in candidates:
        if len(chosen) >= config.WIKI_TOP_PASSAGES:
            break
        # Не отправляем один и тот же текст несколько раз из разных выдач.
        if paragraph in seen_passages:
            continue
        entry = f"[{title}] {paragraph}"
        separator_len = 2 if chosen else 0
        if total_len + separator_len + len(entry) > config.WIKI_MAX_CHARS:
            continue
        chosen.append(entry)
        seen_passages.add(paragraph)
        total_len += separator_len + len(entry)
        if title not in used_titles:
            used_titles.append(title)


    sources = [
        {"title": t, "url": f"https://{lang}.wikipedia.org/wiki/{t.replace(' ', '_')}"}
        for t in used_titles
    ]
    return {"text": "\n\n".join(chosen), "sources": sources}