"""
test_agents.py — проверка этапа 3: каждый агент делает свою работу.

Запуск:  python test_agents.py

Здесь мы вручную, шаг за шагом, проводим ответ через всех агентов,
чтобы увидеть, что делает каждый. На этапе 4 эта цепочка будет
собрана в один модуль (pipeline.py).

Расход лимита Groq: примерно 12–16 запросов (~10 000 токенов).
"""

from agents import Editor, Extractor, Generator, Verifier, VERDICT_ICONS
from llm import LLM
from wiki_search import find_evidence


def line(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def check_answer(llm: LLM, question: str, answer: str):
    """Провести готовый ответ через экстрактор, поиск, верификатор и редактор."""
    print("Вопрос:", question)
    print("Ответ с ошибкой:", answer)

    # --- Экстрактор ---
    claims = Extractor(llm).run(answer, question)
    print(f"\n[ЭКСТРАКТОР] Найдено фактов: {len(claims)}")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c['claim']}")
        print(f"     запросы: {c['queries']}")

    # --- Поиск + верификатор ---
    print("\n[ПОИСК + ВЕРИФИКАТОР]")
    verifier = Verifier(llm)
    checked = []
    for i, c in enumerate(claims, 1):
        evidence = find_evidence(c["claim"], query=c["queries"])
        result = verifier.run(c["claim"], evidence["text"])
        checked.append({**c, **result})

        print(f"\n{i}. {VERDICT_ICONS[result['verdict']]} {result['verdict']}: {c['claim']}")
        print(f"   Пояснение: {result['reasoning']}")
        if result["correction"]:
            print(f"   Исправление: {result['correction']}")
        if evidence["sources"]:
            print("   Источники: " + ", ".join(s["title"] for s in evidence["sources"]))

    # --- Редактор ---
    fixed = Editor(llm).run(question, answer, checked)
    print("\n[РЕДАКТОР]")
    print("Было:  ", answer)
    print("Стало: ", fixed)


def main():
    llm = LLM()

    # -----------------------------------------------------------------------
    # Часть 1. Генератор: просто отвечает на вопрос
    # -----------------------------------------------------------------------
    line("ЧАСТЬ 1. ГЕНЕРАТОР")
    question_ru = "Кто такой Александр Грейам Белл?"
    print("Вопрос:", question_ru)
    print("Ответ: ", Generator(llm).run(question_ru))

    # -----------------------------------------------------------------------
    # Часть 2. Английский ответ с заранее вставленной ошибкой:
    # Белл родился в Эдинбурге, а не в Лондоне.
    # Факт про AT&T правдивый — проверяем, что улучшенный поиск его подтвердит.
    # -----------------------------------------------------------------------
    line("ЧАСТЬ 2. ПРОВЕРКА АНГЛИЙСКОГО ОТВЕТА")
    check_answer(
        llm,
        question="Who was Alexander Graham Bell?",
        answer=(
            "Alexander Graham Bell was a Scottish-born inventor who is credited with "
            "patenting the first practical telephone. He was born in London in 1847. "
            "He co-founded the American Telephone and Telegraph Company (AT&T) in 1885."
        ),
    )

    # -----------------------------------------------------------------------
    # Часть 3. Русский ответ с ошибкой:
    # Пушкин родился в Москве, а не в Санкт-Петербурге.
    # Проверяем, что вся цепочка работает по-русски.
    # -----------------------------------------------------------------------
    line("ЧАСТЬ 3. ПРОВЕРКА РУССКОГО ОТВЕТА")
    check_answer(
        llm,
        question="Кто такой Пушкин?",
        answer=(
            "Александр Сергеевич Пушкин — русский поэт и писатель. "
            "Он родился в Санкт-Петербурге в 1799 году. "
            "Пушкин написал роман в стихах «Евгений Онегин»."
        ),
    )

    line()
    print(
        f"Запросов к модели: {llm.stats['requests']}, "
        f"токенов: {llm.stats['tokens']}, "
        f"время: {llm.stats['seconds']:.1f} с"
    )
    print("Готово! Ожидаем:")
    print("  - часть 2: Лондон ❌, AT&T теперь ✅ (благодаря поиску по нескольким запросам);")
    print("  - часть 3: Санкт-Петербург ❌, пояснения на русском, редактор исправил на Москву.")


if __name__ == "__main__":
    main()