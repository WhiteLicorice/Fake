import unittest

from pydantic import ValidationError

from app import MODEL_ID, VERSION, News, app, check_news, health_check


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, articles):
        if len(articles) != 1:
            raise AssertionError("Expected one article")
        return [self.prediction]


class ApiContractTests(unittest.TestCase):
    def test_health_response_identifies_ready_model(self):
        self.assertEqual(
            health_check(),
            {"health": "ready", "model": MODEL_ID, "version": VERSION},
        )

    def test_health_endpoint_is_available_for_render(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)

    def test_fake_prediction_preserves_boolean_status_contract(self):
        app.state.ml_model = StubModel(0)
        response = check_news(News(news_body="Ito ay sapat na mahabang artikulo."))

        self.assertTrue(response.status)
        self.assertEqual(response.label, "Fake")

    def test_real_prediction_preserves_boolean_status_contract(self):
        app.state.ml_model = StubModel(1)
        response = check_news(News(news_body="Ito ay sapat na mahabang artikulo."))

        self.assertFalse(response.status)
        self.assertEqual(response.label, "Real")

    def test_short_article_is_rejected(self):
        with self.assertRaises(ValidationError):
            News(news_body="Masyadong maikli")


if __name__ == "__main__":
    unittest.main()
