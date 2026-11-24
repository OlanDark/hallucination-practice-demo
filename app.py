"""Локальный веб-интерфейс демонстрационной системы проверки фактов.

Запуск из папки проекта:
    python -m streamlit run app.py

Интерфейс не реализует проверку самостоятельно: он вызывает pipeline.py.
"""

from __future__ import annotations

import html
import json
from difflib import SequenceMatcher
from urllib.parse import urlparse

import streamlit as st

import config
from agents import NOT_ENOUGH_INFO, REFUTED, SUPPORTED
from llm import usage_today
from pipeline import HallucinationPipeline, PipelineResult, format_report


# Примеры для живой демонстрации. Во втором режиме ошибки заданы намеренно:
# это позволяет продемонстрировать проверку, даже если модель отвечает верно.
QUESTIONS = {
    "Белл": "Кто такой Александр Грейам Белл? Где он родился и умер, чем известен?",
    "Гагарин": "Кто такой Юрий Гагарин? Когда и где он совершил свой первый космический полёт?",
    "Эйфелева башня": "Где находится Эйфелева башня, кто её построил и в каком году?",
}

TEXT_EXAMPLES = {
    "Белл": (
        "Александр Грейам Белл — изобретатель, один из создателей телефона. "
        "Он родился в Лондоне в 1847 году."
    ),
    "Пушкин": (
        "Александр Сергеевич Пушкин — русский поэт. "
        "Он родился в Санкт-Петербурге в 1799 году. "
        "Пушкин написал роман в стихах «Евгений Онегин»."
    ),
    "Эйфелева башня": (
        "Эйфелева башня находится в Берлине. "
        "Её построили к Всемирной выставке 1889 года."
    ),
}


@st.cache_data(show_spinner=False)
def _model_options(provider: str) -> list[str]:
    """Простой список моделей, не требующий запроса к облачному API."""
    base = config.DEFAULT_MODELS[provider]
    if provider == "groq":
        return list(dict.fromkeys([base, "openai/gpt-oss-120b"]))
    return [base]


def _safe_wikipedia_url(url: str) -> bool:
    """В интерфейсе показываем переходы только на страницы Wikipedia по HTTPS."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        return parsed.scheme == "https" and (
            hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")
        )
    except (ValueError, TypeError):
        return False


def _highlight_changes(original: str, corrected: str) -> str:
    """Безопасная HTML-подсветка замен: весь текст обязательно экранируется."""
    before = original.split()
    after = corrected.split()
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    parts: list[str] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        old = html.escape(" ".join(before[i1:i2]))
        new = html.escape(" ".join(after[j1:j2]))
        if opcode == "equal":
            parts.append(old)
        elif opcode == "delete":
            parts.append(f'<del class="removed">{old}</del>')
        elif opcode == "insert":
            parts.append(f'<ins class="added">{new}</ins>')
        elif opcode == "replace":
            parts.append(f'<del class="removed">{old}</del>')
            parts.append(f'<ins class="added">{new}</ins>')
    return " ".join(part for part in parts if part)


def _configure_sidebar() -> None:
    st.sidebar.header("Настройки")
    options = _model_options(config.PROVIDER)
    active_model = config.AGENT_MODELS["generator"]
    selected_model = st.sidebar.selectbox(
        "Языковая модель",
        options,
        index=options.index(active_model) if active_model in options else 0,
        help="Одна модель используется для всех четырёх агентов.",
    )
    # Применяем настройку перед созданием HallucinationPipeline.
    for role in config.AGENT_MODELS:
        config.AGENT_MODELS[role] = selected_model

    config.WIKI_MULTI_QUERY = st.sidebar.checkbox(
        "Поиск по нескольким запросам",
        value=config.WIKI_MULTI_QUERY,
        help="Дополнительно ищет статьи по разным названиям сущностей.",
    )
    ranking_names = {"bm25": "BM25", "overlap": "Простой подсчёт слов"}
    ranking = st.sidebar.radio(
        "Отбор абзацев",
        options=list(ranking_names),
        index=0 if config.WIKI_RANKING == "bm25" else 1,
        format_func=lambda code: ranking_names[code],
        help="Определяет, какие найденные абзацы передать верификатору.",
    )
    config.WIKI_RANKING = ranking

    st.sidebar.divider()
    st.sidebar.caption(
        "Это локальный интерфейс. Запросы к модели отправляются через "
        "настроенный сервис; поиск обращается к Википедии."
    )


def _show_limits() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Расход и лимиты")
    usage = usage_today()
    if usage:
        model = config.AGENT_MODELS["generator"]
        current = usage.get(model)
        if current:
            st.sidebar.caption(
                f"По журналу программы сегодня: {current['requests']} запросов, "
                f"{current['tokens']:,} токенов.".replace(",", " ")
            )
        else:
            st.sidebar.caption("Для выбранной модели сегодня записанных запросов нет.")
    else:
        st.sidebar.caption("Журнал запросов пока пуст.")

    limits = st.session_state.get("last_rate_limits", {})
    remaining = limits.get("requests_left_today")
    if remaining is not None:
        st.sidebar.caption(f"Groq: осталось запросов по последнему ответу API — {remaining}.")
    else:
        st.sidebar.caption("Остаток запросов Groq появится после успешного запуска.")
    st.sidebar.caption(
        "Журнал программы не равен точному остатку лимита Groq: "
        "запросы из других приложений в нём не учитываются."
    )


def _render_result(result: PipelineResult) -> None:
    st.divider()
    st.subheader("Исходный ответ")
    st.write(result.answer)

    supported = result.count(SUPPORTED)
    refuted = result.count(REFUTED)
    unknown = result.count(NOT_ENOUGH_INFO)
    st.subheader("Проверка утверждений")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Подтверждено", supported)
    col_b.metric("Опровергнуто", refuted)
    col_c.metric("Недостаточно данных", unknown)
    if result.claims and len(result.claims) >= config.EXTRACTOR_MAX_CLAIMS:
        st.caption(
            f"Достигнут лимит извлечения: {config.EXTRACTOR_MAX_CLAIMS} фактов. "
            "В исходном тексте могут быть другие утверждения, которые не проверялись."
        )

    # В pipeline.py пустой список фактов трактуется как False. Не показываем
    # такой результат пользователю как доказательство достоверности.
    if not result.claims:
        st.warning(
            "Проверка не состоялась: система не выделила проверяемых утверждений. "
            "Это не означает, что исходный текст достоверен."
        )
    elif refuted:
        st.error(f"Обнаружено опровергнутых утверждений: {refuted}.")
    elif unknown:
        st.warning(
            "Прямых противоречий в проверенных утверждениях не найдено, "
            "но для части фактов недостаточно данных."
        )
    else:
        st.success(
            "В выделенных и проверенных утверждениях противоречий источникам не обнаружено."
        )

    for number, claim in enumerate(result.claims, start=1):
        headline = f"{number}. {claim.icon} {claim.claim}"
        if claim.verdict == REFUTED:
            st.error(headline)
        elif claim.verdict == SUPPORTED:
            st.success(headline)
        else:
            st.warning(headline)

        if claim.reasoning:
            st.write("**Почему такой вердикт:**", claim.reasoning)
        if claim.verdict == REFUTED and claim.correction:
            st.write("**Предложенное исправление:**", claim.correction)

        with st.expander(f"Источники и доказательства · факт {number}"):
            if claim.sources:
                st.write("**Найденные статьи:**")
                for source in claim.sources:
                    title = str(source.get("title", "Статья Википедии"))
                    url = str(source.get("url", ""))
                    if _safe_wikipedia_url(url):
                        st.link_button(title, url)
                    else:
                        st.write(title)
            else:
                st.caption("Источники не найдены.")
            if claim.evidence:
                st.write("**Текст, переданный верификатору:**")
                st.text(claim.evidence)
            else:
                st.caption("Подтверждающий текст отсутствует.")

    if result.corrected_answer != result.answer:
        st.subheader("Ответ после исправления")
        st.write(result.corrected_answer)
        st.caption("Это итоговый текст, предложенный агентом-редактором.")
        if unknown:
            st.warning(
                "В исправленном ответе могут остаться неподтверждённые утверждения: "
                "агент исправляет только опровергнутые факты."
            )
        with st.expander("Показать сравнение с исходным ответом"):
            st.markdown(
                '<div class="diff-result">'
                + _highlight_changes(result.answer, result.corrected_answer)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Это не второй ответ: зачёркнуто — удалённый исходный текст, "
                "зелёным — добавленный при исправлении."
            )
        st.caption("Исправленный ответ автоматически повторно не проверяется.")
    elif not result.claims:
        st.info("Исправление не выполнялось: не удалось выделить факты для проверки.")
    else:
        st.info("Ответ остался без изменений: редактор не внёс исправлений.")

    st.subheader("Затраты на выполнение")
    stats = result.stats
    columns = st.columns(4)
    columns[0].metric("Запросы к LLM", stats.get("llm_requests", "—"))
    columns[1].metric("Токены", stats.get("tokens", "—"))
    columns[2].metric("Время LLM, с", stats.get("llm_seconds", "—"))
    columns[3].metric("Общее время, с", stats.get("total_seconds", "—"))

    col_json, col_text = st.columns(2)
    with col_json:
        st.download_button(
            "Скачать JSON результата",
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file_name="fact_check_result.json",
            mime="application/json",
        )
    with col_text:
        st.download_button(
            "Скачать текстовый отчёт",
            data=format_report(result),
            file_name="fact_check_report.txt",
            mime="text/plain",
        )
    st.caption(
        "Это демонстрационный прототип. Итог зависит от полноты найденных источников "
        "и качества решений языковой модели; он не гарантирует истинность всех утверждений."
    )


def main() -> None:
    st.set_page_config(
        page_title="Проверка фактов — демо",
        page_icon="🔎",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .diff-result {line-height: 2; padding: 1rem; border: 1px solid #A0A0A044;
                      border-radius: 0.6rem; overflow-wrap: anywhere;}
        .diff-result .removed {background: #ffd8d8; color: #7d1515;
                               text-decoration: line-through; padding: 0.1rem;}
        .diff-result .added {background: #d7f7df; color: #125826;
                             text-decoration: none; padding: 0.1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _configure_sidebar()

    st.title("Проверка фактов в ответах ИИ")
    st.write(
        "Многоагентная система выделяет утверждения, ищет сведения в Википедии, "
        "определяет возможные ошибки и предлагает исправленный вариант ответа."
    )
    mode = st.radio(
        "Режим работы",
        ["Задать вопрос", "Проверить свой текст"],
        horizontal=True,
    )

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = QUESTIONS["Белл"]
    if "answer_input" not in st.session_state:
        st.session_state["answer_input"] = TEXT_EXAMPLES["Пушкин"]
    if "context_input" not in st.session_state:
        st.session_state["context_input"] = ""

    if mode == "Задать вопрос":
        st.caption("Быстрые примеры:")
        cols = st.columns(len(QUESTIONS))
        for column, (name, question) in zip(cols, QUESTIONS.items()):
            with column:
                if st.button(name, key=f"question_example_{name}"):
                    st.session_state["question_input"] = question
        text = st.text_area(
            "Вопрос для языковой модели", key="question_input", height=120
        )
        st.caption("Модель сначала создаст ответ, после чего начнётся проверка его фактов.")
    else:
        st.caption("Примеры содержат специально добавленные ошибки для демонстрации.")
        cols = st.columns(len(TEXT_EXAMPLES))
        for column, (name, example) in zip(cols, TEXT_EXAMPLES.items()):
            with column:
                if st.button(name, key=f"answer_example_{name}"):
                    st.session_state["answer_input"] = example
        text = st.text_area(
            "Текст, который нужно проверить", key="answer_input", height=170
        )
        context_question = st.text_input(
            "Исходный вопрос (необязательно)",
            key="context_input",
            help="Помогает экстрактору понять контекст короткого ответа.",
        )

    if st.button("Проверить", type="primary"):
        if not text.strip():
            st.warning("Введите вопрос или текст для проверки.")
        else:
            # При новом запуске старый результат сразу сбрасывается.
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_mode", None)
            info_area = st.empty()
            progress_area = st.empty()
            progress_area.progress(0.0)

            def on_step(message: str) -> None:
                info_area.info(message)
                if "Генератор" in message:
                    progress_area.progress(0.1)
                elif "Экстрактор" in message:
                    progress_area.progress(0.2)
                elif "Проверяю факт" in message:
                    try:
                        # Сообщение приходит из pipeline.py: «Проверяю факт i из N: ...».
                        words = message.split()
                        current, total = int(words[2]), int(words[4].rstrip(":"))
                        progress_area.progress(0.2 + 0.7 * current / max(total, 1))
                    except (ValueError, IndexError):
                        pass
                elif "Редактор" in message:
                    progress_area.progress(0.95)
                elif message == "Готово":
                    progress_area.progress(1.0)

            try:
                pipeline = HallucinationPipeline()
                if mode == "Задать вопрос":
                    result = pipeline.run(text.strip(), on_step=on_step)
                else:
                    result = pipeline.check(
                        answer=text.strip(),
                        question=context_question.strip() or None,
                        edit=True,
                        on_step=on_step,
                    )
                st.session_state["last_result"] = result
                st.session_state["last_mode"] = mode
                st.session_state["last_rate_limits"] = dict(pipeline.llm.rate_limits)
                info_area.success("Проверка завершена.")
            except Exception as exc:
                progress_area.empty()
                info_area.error("Не удалось завершить проверку.")
                if isinstance(exc, ValueError) and "API-ключ" in str(exc):
                    st.error("Не найден ключ Groq. Убедитесь, что в папке проекта есть .env.")
                elif "429" in str(exc) or "RateLimit" in type(exc).__name__:
                    st.error("Groq ограничил количество запросов. Проверьте лимиты и повторите позже.")
                else:
                    st.error(f"{type(exc).__name__}: {exc}")

    result = st.session_state.get("last_result")
    if result is not None and st.session_state.get("last_mode") == mode:
        _render_result(result)

    _show_limits()


if __name__ == "__main__":
    main()
