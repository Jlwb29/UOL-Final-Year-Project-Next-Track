import time
from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Track


class TestRequestHandling(TestCase):
    def setUp(self):
        self.client = APIClient()

        Track.objects.create(
            track_id="601",
            title="Seed A",
            artist_name="Artist A",
            genre="pop",
            mood="relaxed",
            era="2010s",
            tempo_bpm=120,
            year=2015,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0.50, 0.80]
        )

        Track.objects.create(
            track_id="602",
            title="Seed B",
            artist_name="Artist B",
            genre="pop",
            mood="relaxed",
            era="2010s",
            tempo_bpm=118,
            year=2014,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0] + [0.52, 0.78]
        )

        Track.objects.create(
            track_id="603",
            title="Candidate High",
            artist_name="Artist C",
            genre="pop",
            mood="relaxed",
            era="2000s",
            tempo_bpm=115,
            year=2008,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0] + [0.49, 0.60]
        )

        Track.objects.create(
            track_id="604",
            title="Candidate Low",
            artist_name="Artist D",
            genre="pop",
            mood="energetic",
            era="2020s",
            tempo_bpm=150,
            year=2022,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0, 0, 0, 0] + [0, 0, 0, 0, 1] + [0.90, 0.95]
        )

    def _timed_get(self, path, params=None):
        t0 = time.perf_counter()
        res = self.client.get(path, params or {})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return res, dt_ms

    def test_tracks_list_latency_and_schema(self):
        res, dt_ms = self._timed_get("/tracks")
        print(f"[LATENCY] GET /tracks = {dt_ms:.2f} ms")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertIn("track_id", data[0])
        self.assertIn("title", data[0])
        self.assertIn("artist_name", data[0])
        self.assertLess(dt_ms, 500.0)

    def test_track_detail_latency_and_schema(self):
        res, dt_ms = self._timed_get("/tracks/601")
        print(f"[LATENCY] GET /tracks/601 = {dt_ms:.2f} ms")

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["track_id"], "601")
        self.assertIn("title", data)
        self.assertIn("artist_name", data)
        self.assertLess(dt_ms, 500.0)

    def test_recommendations_latency_schema_and_ranking(self):
        res, dt_ms = self._timed_get(
            "/recommendations",
            {"seed_ids": "601,602", "limit": "5"}
        )
        print(f"[LATENCY] GET /recommendations = {dt_ms:.2f} ms")

        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("requested_seeds", data)
        self.assertIn("used_seeds", data)
        self.assertIn("invalid_seeds", data)
        self.assertIn("seed_summary", data)
        self.assertIn("recommendations", data)
        self.assertIsInstance(data["recommendations"], list)

        recs = data["recommendations"]
        self.assertGreater(len(recs), 0)

        for r in recs:
            self.assertIn("track_id", r)
            self.assertIn("title", r)
            self.assertIn("artist_name", r)
            self.assertIn("similarity", r)

        scores = [float(r["similarity"]) for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

        titles = [r["title"] for r in recs]
        self.assertIn("Candidate High", titles)

        self.assertEqual(data["used_seeds"], ["601", "602"])
        self.assertEqual(data["invalid_seeds"], [])

        self.assertLess(dt_ms, 500.0)