"""Автономные тесты обновлённых агентов. Не вызывают Groq или Википедию.

Запуск из папки проекта:
    python -m unittest test_agents_upgrade -v
"""

import json
import sys
import types
import unittest
from unittest.mock import patch

# При тестировании реальный API-клиент не нужен: все ответы заменены FakeLLM.
# Это также позволяет запускать тесты без установленного пакета openai.
if "llm" not in sys.modules:
    fake_llm_module = types.ModuleType("llm")
    fake_llm_module.LLM = object
    sys.modules["llm"] = fake_llm_module

from agents import Editor, Extractor, Verifier, NOT_ENOUGH_INFO, REFUTED, SUPPORTED


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def ask(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Неожиданный дополнительный вызов LLM")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


BELl_CLAIM = "Александр Грейам Белл известен как изобретатель телефона."
DISPUTED = (
    "[Bell] Александр Грейам Белл известен как изобретатель телефона.\n\n"
    "[History] Некоторые авторы оспаривают, кто первым изобрёл телефон."
)
DIED = "[Bell] Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."


class TestVerifierUpgrade(unittest.TestCase):
    def test_no_evidence_does_not_call_model(self):
        llm = FakeLLM()
        checked = Verifier(llm).run("Белл родился в Москве.", " ")
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(llm.calls, [])

    def test_supported_only_one_llm_request(self):
        llm = FakeLLM({
            "reasoning": "Есть прямое подтверждение.",
            "verdict": SUPPORTED, "correction": "",
            "supporting_quote": "Белл родился в Эдинбурге, Шотландия.",
            "opposing_quote": "",
        })
        checked = Verifier(llm).run("Белл родился в Эдинбурге.", "[Bell] Белл родился в Эдинбурге, Шотландия.")
        self.assertEqual(checked["verdict"], SUPPORTED)
        self.assertEqual(len(llm.calls), 1)

    def test_conflicting_quotes_cannot_be_refuted(self):
        llm = FakeLLM({
            "reasoning": "Есть спор.", "verdict": REFUTED,
            "correction": "Белл не изобретал телефон.",
            "supporting_quote": "Александр Грейам Белл известен как изобретатель телефона.",
            "opposing_quote": "Некоторые авторы оспаривают, кто первым изобрёл телефон.",
        })
        checked = Verifier(llm).run(BELl_CLAIM, DISPUTED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(checked["correction"], "")
        self.assertEqual(len(llm.calls), 1)

    def test_fabricated_quote_is_not_sufficient_for_refutation(self):
        llm = FakeLLM({
            "reasoning": "Ошибка.", "verdict": REFUTED,
            "correction": "Белл умер в Баддеке.",
            "supporting_quote": "",
            "opposing_quote": "Белл умер на Марсе, а не в Ванкувере.",
        })
        checked = Verifier(llm).run("Белл умер в Ванкувере.", DIED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(len(llm.calls), 2)

    def test_valid_direct_refutation_needs_audit(self):
        quote = "Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."
        llm = FakeLLM(
            {"reasoning": "He died in Baddeck, not Vancouver.", "verdict": REFUTED,
             "correction": "Bell died in Baddeck, Canada.",
             "supporting_quote": "", "opposing_quote": quote},
            {"direct_refutation": True, "supporting_evidence": False,
             "supporting_quote": "", "opposing_quote": quote},
        )
        checked = Verifier(llm).run("Bell died in Vancouver, Canada.", DIED)
        self.assertEqual(checked["verdict"], REFUTED)
        self.assertEqual(checked["correction"], "Bell died in Baddeck, Canada.")
        self.assertEqual(len(llm.calls), 2)

    def test_second_review_blocks_disputed_red(self):
        llm = FakeLLM(
            {"reasoning": "Спорный факт.", "verdict": REFUTED,
             "correction": "Белл не изобретал телефон.", "supporting_quote": "",
             "opposing_quote": "Некоторые авторы оспаривают, кто первым изобрёл телефон."},
            {"direct_refutation": False, "supporting_evidence": True,
             "supporting_quote": "Александр Грейам Белл известен как изобретатель телефона.",
             "opposing_quote": "Некоторые авторы оспаривают, кто первым изобрёл телефон."},
        )
        checked = Verifier(llm).run(BELl_CLAIM, DISPUTED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(checked["correction"], "")
        self.assertEqual(len(llm.calls), 2)

    def test_missing_audit_json_blocks_red(self):
        quote = "Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."
        llm = FakeLLM(
            {"reasoning": "No", "verdict": REFUTED,
             "correction": "Bell died in Baddeck.",
             "supporting_quote": "", "opposing_quote": quote},
            "not json",
        )
        checked = Verifier(llm).run("Bell died in Vancouver.", DIED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)

    def test_second_review_api_error_becomes_yellow(self):
        quote = "Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."
        llm = FakeLLM(
            {"reasoning": "Different place", "verdict": REFUTED,
             "correction": "Bell died in Baddeck.", "supporting_quote": "", "opposing_quote": quote},
            RuntimeError("simulated API limit"),
        )
        checked = Verifier(llm).run("Bell died in Vancouver.", DIED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(checked["correction"], "")

    def test_refuted_without_correction_is_not_safe(self):
        quote = "Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."
        llm = FakeLLM({"reasoning": "Different place", "verdict": REFUTED,
                       "correction": "", "supporting_quote": "", "opposing_quote": quote})
        checked = Verifier(llm).run("Bell died in Vancouver.", DIED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(len(llm.calls), 2)

    def test_missing_first_quote_and_correction_recovered(self):
        # Этот случай воспроизводит скриншот: первая LLM распознала красный факт,
        # но не приложила цитату и исправление, хотя доказательство найдено.
        evidence = (
            "[Белл] Александр Грейам Белл умер 2 августа 1922 года "
            "в Баддеке, провинция Новая Шотландия, Канада."
        )
        quote = "Александр Грейам Белл умер 2 августа 1922 года в Баддеке"
        llm = FakeLLM(
            {"verdict": REFUTED, "reasoning": "Место смерти не совпадает.",
             "correction": "", "supporting_quote": "", "opposing_quote": ""},
            {"direct_refutation": True, "supporting_evidence": False,
             "supporting_quote": "", "opposing_quote": quote,
             "correction": "Александр Грейам Белл умер в Баддеке, Канада."},
        )
        checked = Verifier(llm).run("Александр Грейам Белл умер в Ванкувере, Канада.", evidence)
        self.assertEqual(checked["verdict"], REFUTED)
        self.assertEqual(checked["correction"], "Александр Грейам Белл умер в Баддеке, Канада.")
        self.assertIn(quote, checked["reasoning"])
        self.assertEqual(len(llm.calls), 2)

    def test_missing_initial_quote_disputed_stays_yellow(self):
        llm = FakeLLM(
            {"verdict": REFUTED, "reasoning": "Источники не согласны.",
             "correction": "", "supporting_quote": "", "opposing_quote": ""},
            {"direct_refutation": False, "supporting_evidence": True,
             "supporting_quote": "Александр Грейам Белл известен как изобретатель телефона.",
             "opposing_quote": "Некоторые авторы оспаривают, кто первым изобрёл телефон.",
             "correction": ""},
        )
        checked = Verifier(llm).run(BELl_CLAIM, DISPUTED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(checked["correction"], "")
        self.assertEqual(len(llm.calls), 2)

    def test_review_cannot_use_fabricated_quote(self):
        llm = FakeLLM(
            {"verdict": REFUTED, "reasoning": "Место неверно.",
             "correction": "", "opposing_quote": ""},
            {"direct_refutation": True, "supporting_evidence": False,
             "supporting_quote": "", "opposing_quote": "Bell actually died on Mars.",
             "correction": "Bell died in Baddeck."},
        )
        checked = Verifier(llm).run("Bell died in Vancouver.", DIED)
        self.assertEqual(checked["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(checked["correction"], "")
        self.assertEqual(len(llm.calls), 2)


class TestBiographicalEvidenceFocus(unittest.TestCase):
    """Регрессии по реальным фрагментам со скриншотов пользователя.

    Содержимое ответов FakeLLM задано вручную: тесты не доказывают, что
    Groq даст именно такие ответы на реальном запуске.
    """

    @staticmethod
    def mixed_evidence():
        return (
            "[Белл, Александр Грейам] Александp Грейам Белл "
            "(3 марта 1847, Эдинбург, Шотландия — 2 августа 1922, "
            "Баддек, провинция Новая Шотландия, Канада) — изобретатель.\n\n"
            "[Белл, Александр Мелвилл] Александр Мелвилл Белл "
            "умер 7 августа 1905 года.\n\n"
            "[Ванкувер] Ванкувер — город в Канаде.\n\n"
            "[Ванкувер Кэнакс] Хоккейный клуб из Ванкувера.\n\n"
            "[Белл, Александр Грейам] Белл скончался 2 августа 1922 года "
            "в своем поместье близ города Баддек."
        ).replace("\\n", "\n")

    def test_biography_filter_excludes_city_and_relatives(self):
        from agents import _focus_biography_evidence
        evidence = self.mixed_evidence()
        focused = _focus_biography_evidence(
            "Александр Грейам Белл умер 2 августа 1922 года в Ванкувере.", evidence
        )
        self.assertIn("Баддек", focused)
        self.assertNotIn("[Ванкувер]", focused)
        self.assertNotIn("[Белл, Александр Мелвилл]", focused)
        self.assertEqual(focused.count("[Белл, Александр Грейам]"), 2)

    def test_initial_yellow_biography_rechecked_and_refuted(self):
        evidence = self.mixed_evidence()
        quote = "Белл скончался 2 августа 1922 года в своем поместье близ города Баддек."
        llm = FakeLLM(
            {"verdict": NOT_ENOUGH_INFO, "reasoning": "Сведения неоднозначны.",
             "supporting_quote": "Ванкувер — город в Канаде.",
             "opposing_quote": "", "correction": ""},
            {"direct_refutation": True, "supporting_evidence": False,
             "opposing_quote": quote, "supporting_quote": "",
             "correction": "Александр Грейам Белл умер 2 августа 1922 года в Баддеке, Канада."},
        )
        claim = "Александр Грейам Белл умер 2 августа 1922 года в Ванкувере, Канада."
        result = Verifier(llm).run(claim, evidence)
        self.assertEqual(result["verdict"], REFUTED)
        self.assertIn("Баддеке", result["correction"])
        self.assertEqual(len(llm.calls), 2)
        for call in llm.calls:
            self.assertNotIn("[Ванкувер]", call["prompt"])
            self.assertNotIn("[Белл, Александр Мелвилл]", call["prompt"])

    def test_biography_false_support_quote_does_not_prevent_audit(self):
        evidence = self.mixed_evidence()
        quote = "Белл скончался 2 августа 1922 года в своем поместье близ города Баддек."
        llm = FakeLLM(
            {"verdict": REFUTED, "reasoning": "Место неверное.",
             "supporting_quote": "Ванкувер — город в Канаде.",
             "correction": "", "opposing_quote": quote},
            {"direct_refutation": True, "supporting_evidence": False,
             "opposing_quote": quote, "supporting_quote": "",
             "correction": "Александр Грейам Белл умер в Баддеке, Канада."},
        )
        result = Verifier(llm).run(
            "Александр Грейам Белл умер в Ванкувере, Канада.", evidence)
        self.assertEqual(result["verdict"], REFUTED)

    def test_biography_ambiguous_second_pass_remains_yellow(self):
        evidence = self.mixed_evidence()
        llm = FakeLLM(
            {"verdict": NOT_ENOUGH_INFO, "reasoning": "Нет данных."},
            {"direct_refutation": False, "supporting_evidence": False,
             "opposing_quote": "", "supporting_quote": "", "correction": ""},
        )
        result = Verifier(llm).run(
            "Александр Грейам Белл умер в Ванкувере, Канада.", evidence)
        self.assertEqual(result["verdict"], NOT_ENOUGH_INFO)

    def test_other_yellow_claim_does_not_require_extra_call(self):
        llm = FakeLLM({"verdict": NOT_ENOUGH_INFO, "reasoning": "Нет данных."})
        result = Verifier(llm).run(
            "Александр Грейам Белл преподавал математику.", self.mixed_evidence())
        self.assertEqual(result["verdict"], NOT_ENOUGH_INFO)
        self.assertEqual(len(llm.calls), 1)


class TestExtractorAndEditorUpgrade(unittest.TestCase):
    def test_extractor_keeps_protocol_and_atomic_prompt(self):
        llm = FakeLLM({"claims": [
            {"claim": "Белл был изобретателем.", "queries": ["Александр Грейам Белл"]},
            {"claim": "Белл был педагогом.", "queries": ["Александр Грейам Белл"]},
        ]})
        claims = Extractor(llm).run("Белл был изобретателем и педагогом.")
        self.assertEqual(len(claims), 2)
        self.assertIn("ONE proposition", llm.calls[0]["system"])
        self.assertEqual(claims[1]["queries"], ["Александр Грейам Белл"])

    def test_editor_ignores_yellow_and_green(self):
        llm = FakeLLM()
        answer = "Белл известен как изобретатель телефона."
        checked = Editor(llm).run("Кто такой Белл?", answer, [
            {"claim": "Белл был педагогом.", "verdict": NOT_ENOUGH_INFO, "correction": ""},
            {"claim": "Белл известен как изобретатель телефона.", "verdict": SUPPORTED, "correction": ""},
            {"claim": "Белл был первым.", "verdict": REFUTED, "correction": ""},
        ])
        self.assertEqual(checked, answer)
        self.assertEqual(llm.calls, [])

    def test_editor_applies_only_refuted_corrections(self):
        llm = FakeLLM("Белл известен как изобретатель телефона и умер в Баддеке.")
        answer = "Белл известен как изобретатель телефона и умер в Ванкувере."
        updated = Editor(llm).run("Кто такой Белл?", answer, [
            {"claim": "Белл известен как изобретатель телефона.", "verdict": SUPPORTED, "correction": ""},
            {"claim": "Белл умер в Ванкувере.", "verdict": REFUTED,
             "correction": "Белл умер в Баддеке."},
        ])
        self.assertIn("умер в Баддеке", updated)
        self.assertIn("изобретатель телефона", updated)
        self.assertNotIn("изобретатель телефона", llm.calls[0]["prompt"].split("CONFIRMED WRONG FACTS AND CORRECTIONS:")[1])

    def test_editor_rejects_duplicated_answer(self):
        answer = "Белл родился в Эдинбурге и умер в Ванкувере. " * 3
        llm = FakeLLM(answer + answer + answer)
        updated = Editor(llm).run("", answer, [
            {"claim": "Белл умер в Ванкувере.", "verdict": REFUTED,
             "correction": "Белл умер в Баддеке."},
        ])
        self.assertEqual(updated, answer)


class TestPipelineCompatibility(unittest.TestCase):
    def test_full_check_keeps_disputed_fact_and_corrects_place(self):
        from pipeline import HallucinationPipeline

        quote = "Alexander Graham Bell died in Baddeck, Nova Scotia, Canada in 1922."
        original = (
            "Александр Грейам Белл известен как изобретатель телефона. "
            "Александр Грейам Белл умер в Ванкувере."
        )
        corrected = (
            "Александр Грейам Белл известен как изобретатель телефона. "
            "Александр Грейам Белл умер в Баддеке."
        )

        class CountedFakeLLM(FakeLLM):
            def __init__(self, *responses):
                super().__init__(*responses)
                self.stats = {"requests": 0, "tokens": 0, "seconds": 0.0}
            def ask(self, **kwargs):
                value = super().ask(**kwargs)
                self.stats["requests"] += 1
                self.stats["tokens"] += 100
                return value

        llm = CountedFakeLLM(
            # Извлечение двух фактов.
            {"claims": [
                {"claim": BELl_CLAIM, "queries": ["Александр Грейам Белл"]},
                {"claim": "Александр Грейам Белл умер в Ванкувере.",
                 "queries": ["Александр Грейам Белл"]},
            ]},
            # Спорное авторство -> жёлтый.
            {"reasoning": "Источники обсуждают вопрос приоритета.",
             "verdict": NOT_ENOUGH_INFO, "correction": "",
             "supporting_quote": "Александр Грейам Белл известен как изобретатель телефона.",
             "opposing_quote": "Некоторые авторы оспаривают, кто первым изобрёл телефон."},
            # Неверное место смерти -> красный.
            {"reasoning": "Умер в Баддеке.", "verdict": REFUTED,
             "correction": "Александр Грейам Белл умер в Баддеке.",
             "supporting_quote": "", "opposing_quote": quote},
            # Проверка красного.
            {"direct_refutation": True, "supporting_evidence": False,
             "supporting_quote": "", "opposing_quote": quote},
            # Редактор.
            corrected,
        )
        def evidence_for_claim(claim, query):
            if "изобретателем телефона" in claim:
                return {"text": DISPUTED, "sources": []}
            return {"text": DIED, "sources": []}

        with patch("pipeline.find_evidence", side_effect=evidence_for_claim):
            result = HallucinationPipeline(llm).check(original)
        self.assertEqual(result.claims[0].verdict, NOT_ENOUGH_INFO)
        self.assertEqual(result.claims[1].verdict, REFUTED)
        self.assertEqual(result.corrected_answer, corrected)
        self.assertEqual(result.stats["llm_requests"], 5)



    def test_pipeline_corrects_wrong_death_place_when_first_quote_is_missing(self):
        from pipeline import HallucinationPipeline
        original = "Александр Грейам Белл умер 2 августа 1922 года в Ванкувере."
        corrected = "Александр Грейам Белл умер 2 августа 1922 года в Баддеке."
        evidence_text = (
            "[Белл, Александр Грейам] Александр Грейам Белл умер 2 августа 1922 года "
            "в Баддеке, провинция Новая Шотландия, Канада."
        )
        quote = "Александр Грейам Белл умер 2 августа 1922 года в Баддеке"

        class CountedFakeLLM(FakeLLM):
            def __init__(self, *responses):
                super().__init__(*responses)
                self.stats = {"requests": 0, "tokens": 0, "seconds": 0.0}
            def ask(self, **kwargs):
                result = super().ask(**kwargs)
                self.stats["requests"] += 1
                self.stats["tokens"] += 120
                return result

        llm = CountedFakeLLM(
            {"claims": [{"claim": original, "queries": ["Александр Грейам Белл"]}]},
            {"verdict": REFUTED, "reasoning": "Место смерти неверное.",
             "correction": "", "supporting_quote": "", "opposing_quote": ""},
            {"direct_refutation": True, "supporting_evidence": False,
             "supporting_quote": "", "opposing_quote": quote,
             "correction": corrected},
            corrected,
        )
        with patch("pipeline.find_evidence", return_value={"text": evidence_text, "sources": []}):
            result = HallucinationPipeline(llm).check(original)
        self.assertEqual(result.count(REFUTED), 1)
        self.assertEqual(result.corrected_answer, corrected)
        self.assertEqual(result.stats["llm_requests"], 4)


if __name__ == '__main__':
    unittest.main()
