"""
pipeline.py — оркестратор: соединяет всех агентов в одну цепочку.

Именно здесь агенты «общаются»: программа берёт результат одного агента
и передаёт его следующему.

    вопрос ─► Генератор ─► ответ ─► Экстрактор ─► факты
                                                    │
                    для каждого факта: Поиск в Википедии ─► Верификатор
                                                    │
                             вердикты ─► Редактор ─► исправленный ответ

Два способа запуска:
  pipeline.run(question)          — полный цикл: модель отвечает на вопрос,
                                    потом сама же проверяет свой ответ
                                    (так работает окно демо);
  pipeline.check(answer, question) — проверить ГОТОВЫЙ ответ, без генератора
                                    (так работает эксперимент на датасете,
                                    где ответы уже даны).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import config
from agents import (
    Editor, Extractor, Generator, Verifier,
    NOT_ENOUGH_INFO, REFUTED, SUPPORTED, VERDICT_ICONS,
)
from llm import LLM
from wiki_search import find_evidence


# ---------------------------------------------------------------------------
# Структуры для хранения результата
# (dataclass — просто удобная «карточка» с именованными полями)
# ---------------------------------------------------------------------------

@dataclass
class CheckedClaim:
    """Один проверенный факт."""
    claim: str                    # текст факта
    queries: list[str]            # поисковые запросы
    verdict: str                  # SUPPORTED / REFUTED / NOT_ENOUGH_INFO
    reasoning: str                # пояснение верификатора
    correction: str               # правильный факт (если опровергнут)
    sources: list[dict]           # статьи Википедии [{"title", "url"}]
    evidence: str                 # текст, по которому проверяли

    @property
    def icon(self) -> str:
        return VERDICT_ICONS.get(self.verdict, "?")


@dataclass
class PipelineResult:
    """Полный результат работы системы."""
    question: str | None
    answer: str                                   # исходный ответ
    claims: list[CheckedClaim] = field(default_factory=list)
    corrected_answer: str = ""                    # ответ после редактора
    stats: dict = field(default_factory=dict)     # запросы, токены, время

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.claims if c.verdict == verdict)

    @property
    def has_hallucination(self) -> bool:
        """
        Итоговое решение: есть ли в ответе галлюцинация.

        Правило: если хотя бы один факт опровергнут — да.
        Если в config.py включено NEI_COUNTS_AS_HALLUCINATION, то факты
        «недостаточно данных» тоже считаются подозрительными.
        Это правило — одна из настроек, которую стоит сравнить в эксперименте.
        """
        if self.count(REFUTED) > 0:
            return True
        if config.NEI_COUNTS_AS_HALLUCINATION and self.count(NOT_ENOUGH_INFO) > 0:
            return True
        return False

    def to_dict(self) -> dict:
        """Превратить результат в словарь (для сохранения в файл)."""
        data = asdict(self)
        data["has_hallucination"] = self.has_hallucination
        return data


# ---------------------------------------------------------------------------
# Сам оркестратор
# ---------------------------------------------------------------------------

class HallucinationPipeline:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or LLM()
        self.generator = Generator(self.llm)
        self.extractor = Extractor(self.llm)
        self.verifier = Verifier(self.llm)
        self.editor = Editor(self.llm)

    # --- Полный цикл: ответить и проверить ---------------------------------
    def run(
        self,
        question: str,
        on_step: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """
        on_step — необязательная функция, которой сообщается текущий шаг
        (окно демо будет показывать его пользователю: «Ищу в Википедии...»).
        """
        notify = on_step or (lambda message: None)
        start, before = time.time(), dict(self.llm.stats)

        notify("Генератор отвечает на вопрос...")
        answer = self.generator.run(question)

        result = self._check(answer, question, edit=True, notify=notify)
        result.stats = self._stats(before, start)
        return result

    # --- Проверка готового ответа ------------------------------------------
    def check(
        self,
        answer: str,
        question: str | None = None,
        edit: bool = True,
        on_step: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """
        edit=False — не вызывать редактора (в эксперименте исправление
        не нужно, важен только вердикт; так экономим лимит).
        """
        notify = on_step or (lambda message: None)
        start, before = time.time(), dict(self.llm.stats)

        result = self._check(answer, question, edit=edit, notify=notify)
        result.stats = self._stats(before, start)
        return result

    # --- Внутренняя логика ---------------------------------------------------
    def _check(self, answer, question, edit, notify) -> PipelineResult:
        result = PipelineResult(question=question, answer=answer)

        notify("Экстрактор разбивает ответ на факты...")
        claims = self.extractor.run(answer, question)

        for i, item in enumerate(claims, 1):
            notify(f"Проверяю факт {i} из {len(claims)}: {item['claim']}")
            evidence = find_evidence(item["claim"], query=item["queries"])
            verdict = self.verifier.run(item["claim"], evidence["text"])
            result.claims.append(CheckedClaim(
                claim=item["claim"],
                queries=item["queries"],
                verdict=verdict["verdict"],
                reasoning=verdict["reasoning"],
                correction=verdict["correction"],
                sources=evidence["sources"],
                evidence=evidence["text"],
            ))

        if edit and result.count(REFUTED) > 0:
            notify("Редактор исправляет ответ...")
            result.corrected_answer = self.editor.run(
                question or "", answer,
                [{"claim": c.claim, "verdict": c.verdict, "correction": c.correction}
                 for c in result.claims],
            )
        else:
            result.corrected_answer = answer

        notify("Готово")
        return result

    def _stats(self, before: dict, start: float) -> dict:
        """Сколько потрачено именно на этот запуск."""
        return {
            "llm_requests": self.llm.stats["requests"] - before["requests"],
            "tokens": self.llm.stats["tokens"] - before["tokens"],
            "llm_seconds": round(self.llm.stats["seconds"] - before["seconds"], 2),
            "total_seconds": round(time.time() - start, 2),  # вместе с Википедией
        }


# ---------------------------------------------------------------------------
# Красивый вывод результата в консоль
# ---------------------------------------------------------------------------

def format_report(result: PipelineResult) -> str:
    lines = []
    if result.question:
        lines.append(f"ВОПРОС: {result.question}")
    lines.append(f"\nОТВЕТ МОДЕЛИ:\n{result.answer}")

    lines.append(f"\nПРОВЕРКА ФАКТОВ ({len(result.claims)}):")
    for i, c in enumerate(result.claims, 1):
        lines.append(f"\n{i}. {c.icon} {c.claim}")
        if c.reasoning:
            lines.append(f"   Пояснение: {c.reasoning}")
        if c.correction:
            lines.append(f"   Исправление: {c.correction}")
        if c.sources:
            lines.append(f"   Источник: {c.sources[0]['url']}")

    lines.append(
        f"\nИТОГ: ✅ {result.count(SUPPORTED)}  "
        f"❌ {result.count(REFUTED)}  "
        f"⚠️ {result.count(NOT_ENOUGH_INFO)}  →  "
        + ("ГАЛЛЮЦИНАЦИЯ ОБНАРУЖЕНА" if result.has_hallucination else "галлюцинаций не найдено")
    )
    if result.corrected_answer != result.answer:
        lines.append(f"\nИСПРАВЛЕННЫЙ ОТВЕТ:\n{result.corrected_answer}")

    s = result.stats
    lines.append(
        f"\nЗатраты: запросов к модели {s.get('llm_requests')}, "
        f"токенов {s.get('tokens')}, "
        f"время модели {s.get('llm_seconds')} с, "
        f"всего {s.get('total_seconds')} с"
    )
    return "\n".join(lines)
