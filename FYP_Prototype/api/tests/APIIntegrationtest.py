from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Track


class TestAPIIntegration(TestCase):
    def setUp(self):
        self.client = APIClient()

        Track.objects.create(
            track_id="501",
            title="Late Night Drive",
            artist_name="Artist A",
            genre="pop",
            mood="relaxed",
            era="2010s",
            tempo_bpm=120,
            year=2015,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0.5, 0.8]
        )

        Track.objects.create(
            track_id="502",
            title="Cozy Evening",
            artist_name="Artist B",
            genre="pop",
            mood="relaxed",
            era="2000s",
            tempo_bpm=115,
            year=2008,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0] + [0.4, 0.6]
        )

    def test_get_tracks_endpoint(self):
        response = self.client.get("/tracks")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)

    def test_get_track_by_id(self):
        response = self.client.get("/tracks/501")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["track_id"], "501")
        self.assertEqual(data["title"], "Late Night Drive")

    def test_recommendations_endpoint(self):
        response = self.client.get(
            "/recommendations",
            {"seed_ids": "501", "limit": "5"}
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)
        self.assertIn("similarity", data["recommendations"][0])