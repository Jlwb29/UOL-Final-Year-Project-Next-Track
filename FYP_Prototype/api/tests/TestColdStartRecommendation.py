from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Track


class TestColdStartRecommendation(TestCase):
    def setUp(self):
        self.client = APIClient()

        Track.objects.create(
            track_id="301",
            title="Late Night Drive",
            artist_name="Artist A",
            genre="pop",
            mood="relaxed",
            era="2010s",
            tempo_bpm=120,
            year=2015,
            feature_vector=[1,0,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0] + [0,0,0,1,0] + [0.50, 0.80]
        )

        Track.objects.create(
            track_id="302",
            title="Midnight Lights",
            artist_name="Artist B",
            genre="pop",
            mood="relaxed",
            era="2010s",
            tempo_bpm=118,
            year=2014,
            feature_vector=[1,0,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0] + [0,0,0,1,0] + [0.48, 0.78]
        )

        Track.objects.create(
            track_id="303",
            title="Cozy Evening",
            artist_name="Artist C",
            genre="pop",
            mood="relaxed",
            era="2000s",
            tempo_bpm=115,
            year=2008,
            feature_vector=[1,0,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0] + [0,0,1,0,0] + [0.40, 0.60]
        )

    def test_recommendations_work_without_user_history(self):
        res = self.client.get("/recommendations", {"seed_ids": "301,302", "limit": "3"})
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)

        recs = data["recommendations"]
        self.assertIn("similarity", recs[0])
        self.assertGreaterEqual(float(recs[0]["similarity"]), 0.0)

        titles = [r["title"] for r in data["recommendations"]]
        self.assertIn("Cozy Evening", titles)