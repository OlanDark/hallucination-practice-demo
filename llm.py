"""
llm.py — обращение к языковой модели.

Это единственное место в программе, которое «разговаривает» с моделью.
Все агенты пользуются классом LLM отсюда: отправляют текст
(промпт) — получают текст (ответ).

Дополнительно класс:
  - считает статистику текущего запуска: запросы, токены, время
    (для таблицы «качество против затрат» в эксперименте);
  - запоминает служебную информацию о лимитах, которую Groq присылает
    вместе с каждым ответом (сколько запросов осталось на сегодня и т.д.);
  - ведёт журнал расхода за день в файле cache/usage_log.json —
    Groq не показывает, сколько токенов осталось на сегодня,
    поэтому считаем сами.
"""

from __future__ import annotations  # чтобы код работал и на старых версиях Python

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import config

USAGE_LOG = Path("cache") / "usage_log.json"


class LLM:
    def __init__(self, provider: str = config.PROVIDER):
        settings = config.PROVIDERS[provider]

        if not settings["api_key"]:
            raise ValueError(
                "Не найден API-ключ. Проверьте, что рядом с программой есть файл .env "
                "и в нём строка GROQ_API_KEY=ваш_ключ"
            )

        self.provider = provider

        # Клиент — объект, который отправляет запросы на сервер.
        # max_retries=5: если сервер ответил «слишком много запросов» (лимит)
        # или временно недоступен, клиент сам подождёт и повторит попытку.
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
            max_retries=5,
            timeout=120,
        )

        self.rate_limits: dict = {}  # последние известные данные о лимитах от Groq
        self.reset_stats()

    def reset_stats(self):
        """Обнулить счётчики запросов, токенов и времени."""
        self.stats = {"requests": 0, "tokens": 0, "seconds": 0.0}

    def ask(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Отправить модели запрос и получить ответ в виде текста.

        prompt      — сам вопрос или задание;
        system      — «системная» инструкция: роль агента
                      (например, «Ты — проверяющий фактов...»);
        model       — какую модель использовать (по умолчанию из config.py);
        temperature — степень «творчества» (0 = предсказуемо).
        """
        # Сообщения в формате чата: сначала роль (system), потом задание (user)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": model or config.DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
        }

        # Для моделей gpt-oss на Groq ограничиваем «рассуждения», чтобы экономить лимит
        if self.provider == "groq" and "gpt-oss" in params["model"]:
            params["reasoning_effort"] = config.REASONING_EFFORT

        start = time.time()
        # with_raw_response — чтобы получить не только ответ, но и служебные
        # заголовки, где Groq сообщает о лимитах
        raw = self.client.chat.completions.with_raw_response.create(**params)
        response = raw.parse()
        elapsed = time.time() - start

        self._remember_rate_limits(raw.headers)

        # Обновляем статистику
        tokens = response.usage.total_tokens if response.usage else 0
        self.stats["requests"] += 1
        self.stats["seconds"] += elapsed
        self.stats["tokens"] += tokens
        self._log_usage(params["model"], tokens)

        answer = response.choices[0].message.content or ""
        return answer.strip()

    # -----------------------------------------------------------------------
    # Учёт лимитов
    # -----------------------------------------------------------------------

    def _remember_rate_limits(self, headers):
        """
        Groq присылает в каждом ответе:
          x-ratelimit-limit-requests / remaining-requests / reset-requests
              — запросы НА ДЕНЬ (лимит, остаток, время до сброса);
          x-ratelimit-limit-tokens / remaining-tokens / reset-tokens
              — токены НА МИНУТУ (дневной остаток токенов Groq не присылает).
        """
        names = {
            "requests_per_day": "x-ratelimit-limit-requests",
            "requests_left_today": "x-ratelimit-remaining-requests",
            "requests_reset_in": "x-ratelimit-reset-requests",
            "tokens_per_minute": "x-ratelimit-limit-tokens",
            "tokens_left_this_minute": "x-ratelimit-remaining-tokens",
            "tokens_reset_in": "x-ratelimit-reset-tokens",
        }
        found = {key: headers.get(header) for key, header in names.items() if headers.get(header)}
        if found:
            self.rate_limits = found

    @staticmethod
    def _log_usage(model: str, tokens: int):
        """Добавить запрос в журнал расхода за день (по дате UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            log = json.loads(USAGE_LOG.read_text(encoding="utf-8")) if USAGE_LOG.exists() else {}
        except json.JSONDecodeError:
            log = {}
        day = log.setdefault(today, {})
        entry = day.setdefault(model, {"requests": 0, "tokens": 0})
        entry["requests"] += 1
        entry["tokens"] += tokens
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        USAGE_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def usage_today() -> dict:
    """Сколько потрачено сегодня (по журналу): {модель: {"requests", "tokens"}}."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not USAGE_LOG.exists():
        return {}
    try:
        return json.loads(USAGE_LOG.read_text(encoding="utf-8")).get(today, {})
    except json.JSONDecodeError:
        return {}