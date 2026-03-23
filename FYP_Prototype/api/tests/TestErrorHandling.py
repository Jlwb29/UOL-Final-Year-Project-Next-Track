from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Track


class TestErrorHandlingAndEdgeCases(TestCase):
    def setUp(self):
        self.client = APIClient()

        Track.objects.create(
            track_id="701",
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
            track_id="702",
            title="Candidate B",
            artist_name="Artist B",
            genre="pop",
            mood="relaxed",
            era="2000s",
            tempo_bpm=115,
            year=2008,
            feature_vector=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 1, 0, 0] + [0.40, 0.60]
        )

    def test_missing_seed_ids_returns_400(self):
        res = self.client.get("/recommendations", {"limit": "5"})
        self.assertEqual(res.status_code, 400)

        data = res.json()
        self.assertIn("non_field_errors", data)
        self.assertIn(
            "Provide at least one seed_id or one preference field.",
            data["non_field_errors"][0]
        )

    def test_non_existent_seed_ids_returns_404(self):
        res = self.client.get("/recommendations", {"seed_ids": "999", "limit": "5"})
        self.assertEqual(res.status_code, 404)

        data = res.json()
        self.assertTrue(
            isinstance(data, dict) and (
                "error" in data or
                "message" in data or
                "missing" in data or
                "detail" in data
            )
        )

    def test_track_detail_non_existent_returns_404(self):
        res = self.client.get("/tracks/999")
        self.assertEqual(res.status_code, 404)

        data = res.json()
        self.assertTrue(
            isinstance(data, dict) and (
                "error" in data or
                "message" in data or
                "detail" in data
            )
        )

    def test_negative_limit_returns_400_with_descriptive_payload(self):
        res = self.client.get("/recommendations", {"seed_ids": "701", "limit": "-5"})
        self.assertEqual(res.status_code, 400)

        data = res.json()
        self.assertTrue(
            isinstance(data, dict) and (
                "limit" in data or
                "detail" in data or
                "message" in data or
                "error" in data or
                "non_field_errors" in data
            )
        )

    def test_large_limit_returns_400_with_descriptive_payload(self):
        res = self.client.get("/recommendations", {"seed_ids": "701", "limit": "999999"})
        self.assertEqual(res.status_code, 400)

        data = res.json()
        self.assertTrue(
            isinstance(data, dict) and (
                "limit" in data or
                "detail" in data or
                "message" in data or
                "error" in data or
                "non_field_errors" in data
            )
        )

    def test_empty_seed_ids_string_returns_400(self):
        res = self.client.get("/recommendations", {"seed_ids": "", "limit": "5"})
        self.assertEqual(res.status_code, 400)

        data = res.json()
        self.assertIn("non_field_errors", data)
        self.assertIn(
            "Provide at least one seed_id or one preference field.",
            data["non_field_errors"][0]
        )