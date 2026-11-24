"""
check_limits.py — сколько лимита Groq потрачено и сколько осталось.

Запуск:  python check_limits.py

Делает ОДИН крошечный запрос к модели (≈ 50–100 токенов), чтобы получить
от Groq свежие данные о лимитах, и показывает журнал расхода за сегодня.
"""

import config
from llm import LLM, usage_today

# Дневной лимит токенов бесплатного тарифа Groq (из документации, ноябрь 2026).
# Сверьте с вашей страницей https://console.groq.com/settings/limits
DAILY_TOKEN_LIMIT = 200_000


def main():
    model = config.DEFAULT_MODEL

    print("=" * 60)
    print(f"Модель: {model}")
    print("=" * 60)

    # 1. Свежие данные от Groq
    llm = LLM()
    llm.ask("Reply with one word: OK", model=model)
    limits = llm.rate_limits

    if limits:
        print("\nДанные от Groq (по последнему запросу):")
        print(f"  Запросов на день:          {limits.get('requests_per_day', '?')}")
        print(f"  Осталось запросов сегодня: {limits.get('requests_left_today', '?')}")
        print(f"  До сброса лимита запросов: {limits.get('requests_reset_in', '?')}")
        print(f"  Токенов в минуту:          {limits.get('tokens_per_minute', '?')}")
        print(f"  Осталось в этой минуте:    {limits.get('tokens_left_this_minute', '?')}")
    else:
        print("\nGroq не прислал данных о лимитах (так бывает с Ollama — у неё лимитов нет).")

    # 2. Наш собственный учёт токенов за день
    usage = usage_today()
    print("\nПотрачено сегодня по журналу (дата по UTC):")
    if not usage:
        print("  пока ничего")
    for model_name, entry in usage.items():
        left = DAILY_TOKEN_LIMIT - entry["tokens"]
        print(
            f"  {model_name}: запросов {entry['requests']}, токенов {entry['tokens']:,}"
            f"  →  осталось примерно {max(left, 0):,} из {DAILY_TOKEN_LIMIT:,}".replace(",", " ")
        )

    print(
        "\nВажно: журнал считает только запросы из этой программы с момента, как он появился.\n"
        "Запросы, сделанные раньше, в нём не учтены. Точные лимиты — на\n"
        "https://console.groq.com/settings/limits"
    )


if __name__ == "__main__":
    main()
