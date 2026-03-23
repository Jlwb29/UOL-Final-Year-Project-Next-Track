import csv
import os
import sys
import django

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "FYP_Prototype.settings")
django.setup()

from api.models import Track

csv_path = os.path.join(BASE_DIR, "tracks.csv")

Track.objects.all().delete()

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        Track.objects.create(
            track_id=row["track_id"],
            title=row["title"],
            artist_name=row["artist_name"],
            genre=row["genre"],
            mood=row["mood"],
            era=row["era"],
            tempo_bpm=int(row["tempo_bpm"]),
            year=int(row["year"]),
            popularity=int(row["popularity"]),
            tags=row.get("tags", "")
        )

print("Tracks successfully loaded.")

