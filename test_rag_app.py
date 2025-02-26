import unittest
from unittest.mock import patch, MagicMock
from rag_app import perform_search

class TestPerformSearch(unittest.TestCase):

    @patch('rag_app.vector_store.client.scroll')
    def test_perform_search_keyword(self, mock_scroll):
        # Mock scroll method
        mock_scroll.side_effect = [
            ([
                MagicMock(payload={'page_content': 'Test document.'}),
                MagicMock(payload={'page_content': 'Test document.'}),
            ], None)
        ]

        results = perform_search('Tset', 'keyword') # Incorrect spelling to test fuzzy

        self.assertEqual(len(results), 2)
        self.assertEqual('Test document.', results[0].page_content)
        self.assertEqual('Test document.', results[1].page_content)

if __name__ == '__main__':
    unittest.main()
