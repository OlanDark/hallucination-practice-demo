"""
test_wiki.py — проверка этапа 2: находит ли программа нужный текст в Википедии.

Запуск:  python test_wiki.py

Модель здесь НЕ используется, лимит Groq не тратится.
"""

from wiki_search import find_evidence

# Факты для проверки: (факт, поисковый запрос).
# Если запрос None — ищем по тексту самого факта.
# В готовой системе поисковый запрос будет формулировать агент-экстрактор.
CLAIMS = [
    ("Alexander Graham Bell was born in London.", None),        # ложь: Эдинбург
    ("The Eiffel Tower was completed in 1889.", None),          # правда
    ("Пушкин родился в Санкт-Петербурге.", None),               # ложь: Москва (неудачный поиск)
    ("Пушкин родился в Санкт-Петербурге.",
     "Александр Сергеевич Пушкин"),                             # то же, но с точным запросом
    ("Роман «Война и мир» написал Лев Толстой.", None),         # правда
]


def main():
    for claim, query in CLAIMS:
        print("=" * 70)
        print("ФАКТ:", claim)
        if query:
            print("ПОИСКОВЫЙ ЗАПРОС:", query)
        print("-" * 70)

        evidence = find_evidence(claim, query=query)

        if not evidence["text"]:
            print("Ничего не найдено.")
            continue

        print("Источники:")
        for source in evidence["sources"]:
            print(f"  - {source['title']}: {source['url']}")
        print()
        print("Найденный текст (первые 800 символов):")
        print(evidence["text"][:800])
        print(f"... (всего {len(evidence['text'])} символов)")

    print("=" * 70)
    print("Готово! Посмотрите, есть ли в найденном тексте информация для проверки каждого факта.")


if __name__ == "__main__":
    main()