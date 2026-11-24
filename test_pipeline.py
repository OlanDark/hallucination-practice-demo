"""
test_pipeline.py — проверка этапа 4: вся система одной командой.

Запуск:  python test_pipeline.py

Здесь ответ с ошибками НЕ подставляется вручную: модель сама отвечает
на вопрос, а система сама проверяет этот ответ. Ошибки, если они будут,
— настоящие галлюцинации модели.

Расход лимита Groq: примерно 8–12 запросов (~8 000 токенов).
"""

from pipeline import HallucinationPipeline, format_report


def main():
    pipeline = HallucinationPipeline()

    question = "Кто такой Александр Грейам Белл? Расскажи о его жизни: где родился и умер, чем известен."

    # on_step=print — печатать каждый шаг, чтобы видеть, что происходит
    result = pipeline.run(question, on_step=lambda step: print(f"  ... {step}"))

    print("\n" + "=" * 70)
    print(format_report(result))
    print("=" * 70)


if __name__ == "__main__":
    main()
