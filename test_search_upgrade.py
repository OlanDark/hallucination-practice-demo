"""Автономные тесты улучшенного поиска. Интернет и ключ Groq НЕ требуются.

Запуск в папке проекта после копирования файлов:
    python -m unittest test_search_upgrade -v
"""
import unittest
from unittest.mock import patch

import config
import wiki_search


class ImprovedWikiSearchTests(unittest.TestCase):
    def test_short_fact_paragraph_is_not_discarded(self):
        sentence = 'Белл умер в Баддеке в Канаде.'
        self.assertIn(sentence, wiki_search._split_paragraphs(sentence))

    def test_long_paragraph_does_not_lose_ending(self):
        tail = 'Александр Белл занимался исследованиями в области авиации.'
        long_text = ('Это длинное описание биографии изобретателя, не говорящее об авиации совсем. ' * 16) + tail
        chunks = wiki_search._split_paragraphs(long_text, max_len=300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(tail in chunk for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))

    def test_search_includes_results_of_multiple_queries(self):
        def fake_search(query, lang):
            return {'Первый':['Статья А1','Статья А2','Статья А3'],
                    'Второй':['Статья Б1','Статья Б2','Статья Б3'],
                    'Сам факт':['Статья В1','Статья В2','Статья В3']}[query]
        article = 'Александр Белл занимался исследованиями в области авиации в течение длительного времени.'
        with (patch.object(wiki_search, '_search_one', side_effect=fake_search),
              patch.object(wiki_search, 'get_article_text', return_value=article),
              patch.object(config, 'WIKI_MULTI_QUERY', True),
              patch.object(config, 'WIKI_MAX_ARTICLES', 3),
              patch.object(config, 'WIKI_TOP_PASSAGES', 3)):
            # Текст одинаковый во всех статьях и дедуплицируется, но статьями
            # действительно должны оказаться А1, Б1, В1.
            wiki_search.find_evidence('Сам факт', ['Первый', 'Второй'], lang='ru')
            calls = [call.args[0] for call in wiki_search.get_article_text.call_args_list]
            self.assertEqual(calls, ['Статья А1', 'Статья Б1', 'Статья В1'])

    def test_ranking_recovers_facts_at_end_of_article(self):
        tail = 'Александр Белл занимался исследованиями в области авиации.'
        article = ('Александр Белл был учёным, работавшим над многими изобретениями. ' * 17) + tail
        with (patch.object(wiki_search, '_search_one', return_value=['Александр Белл']),
              patch.object(wiki_search, 'get_article_text', return_value=article),
              patch.object(config, 'WIKI_MAX_CHARS', 2000),
              patch.object(config, 'WIKI_TOP_PASSAGES', 7)):
            result = wiki_search.find_evidence(tail, query=['Александр Белл'], lang='ru')
            self.assertIn(tail, result['text'])
            self.assertLessEqual(len(result['text']), config.WIKI_MAX_CHARS)

    def test_budget_skips_oversized_candidate(self):
        long_line = 'авиации ' * 24
        short_line = 'Белл исследовал авиацию и разрабатывал летательные аппараты.'
        with (patch.object(wiki_search, '_search_one', return_value=['Название']),
              patch.object(wiki_search, 'get_article_text', return_value=long_line + '\n' + short_line),
              patch.object(config, 'WIKI_MAX_CHARS', 110),
              patch.object(config, 'WIKI_TOP_PASSAGES', 3)):
            result = wiki_search.find_evidence('авиации Белл', query=['Название'], lang='ru')
            self.assertIn(short_line, result['text'])
            self.assertLessEqual(len(result['text']), 110)


if __name__ == '__main__':
    unittest.main()
