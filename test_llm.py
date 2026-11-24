"""
test_llm.py — проверка этапа 1: отвечает ли модель.

Запуск:  python test_llm.py
"""

import config
from llm import LLM


def main():
    print(f"Сервис: {config.PROVIDER}")
    print(f"Модель: {config.DEFAULT_MODEL}")
    print("-" * 50)

    llm = LLM()

    # Тест 1: простой вопрос на английском
    answer = llm.ask("What is the capital of France? Answer with one word.")
    print("Тест 1 (английский):", answer)

    # Тест 2: вопрос на русском
    answer = llm.ask("Кто написал роман «Война и мир»? Ответь одним предложением.")
    print("Тест 2 (русский):  ", answer)

    # Тест 3: системная инструкция — так мы будем задавать роли агентам
    answer = llm.ask(
        prompt="Fact: The Eiffel Tower is located in Berlin.",
        system="You are a fact checker. Reply with exactly one word: TRUE or FALSE.",
    )
    print("Тест 3 (роль):     ", answer)

    print("-" * 50)
    print(
        f"Запросов: {llm.stats['requests']}, "
        f"токенов: {llm.stats['tokens']}, "
        f"время: {llm.stats['seconds']:.1f} с"
    )
    print("Готово! Если вы видите три ответа выше — этап 1 пройден.")


if __name__ == "__main__":
    main()
