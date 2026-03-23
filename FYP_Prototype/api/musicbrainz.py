import os
import sys
import time
import django
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "FYP_Prototype.settings")
django.setup()

from api.models import Track


BASE_URL = "https://musicbrainz.org/ws/2/recording/"
HEADERS = {
    "User-Agent": "NextTrackFYP/1.0 (student project demo)"
}

DEMO_QUERIES = [
    {"title": "Blinding Lights", "artist": "The Weeknd"},
    {"title": "Take On Me", "artist": "a-ha"},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana"},
    {"title": "Fly Me to the Moon", "artist": "Frank Sinatra"},
    {"title": "Summertime", "artist": "Ella Fitzgerald"},
]


def infer_genre_from_tags(tags_text):
    text = str(tags_text).lower()

    if "pop" in text or "synthpop" in text:
        return "pop"
    if "rock" in text or "grunge" in text or "alternative rock" in text:
        return "rock"
    if "hip hop" in text or "hip-hop" in text or "rap" in text:
        return "hiphop"
    if "jazz" in text or "swing" in text or "vocal jazz" in text:
        return "jazz"
    if "classical" in text or "orchestra" in text:
        return "classical"
    if "metal" in text or "heavy metal" in text:
        return "metal"
    if "r&b" in text or "rnb" in text or "soul" in text:
        return "rnb"
    if "indie" in text:
        return "indie"
    if "electronic" in text or "edm" in text or "dance" in text or "house" in text:
        return "edm"

    return "pop"


def infer_mood_from_tags(tags_text):
    text = str(tags_text).lower()

    if "sad" in text or "melancholic" in text or "melancholy" in text:
        return "sad"
    if "happy" in text or "uplifting" in text or "bright" in text:
        return "happy"
    if "relax" in text or "calm" in text or "soft" in text:
        return "relaxed"
    if "energetic" in text or "dance" in text or "party" in text or "upbeat" in text:
        return "energetic"
    if "chill" in text or "ambient" in text:
        return "chill"
    if "romantic" in text or "love" in text or "ballad" in text:
        return "romantic"
    if "angry" in text or "aggressive" in text:
        return "angry"
    if "study" in text or "focus" in text or "instrumental" in text:
        return "focus"

    return "chill"


def infer_era(year):
    if not year:
        return "2020s"

    if 1980 <= year <= 1989:
        return "1980s"
    if 1990 <= year <= 1999:
        return "1990s"
    if 2000 <= year <= 2009:
        return "2000s"
    if 2010 <= year <= 2019:
        return "2010s"

    return "2020s"


def extract_year(first_release_date):
    if not first_release_date:
        return 2020

    try:
        return int(str(first_release_date)[:4])
    except Exception:
        return 2020


def normalize_tag_name(name):
    if not name:
        return ""

    return str(name).strip().lower()


def normalize_text(value):
    if not value:
        return ""

    return str(value).strip().lower()


def search_recording(title, artist):
    query = f'recording:"{title}" AND artist:"{artist}"'
    params = {
        "query": query,
        "fmt": "json",
        "limit": 1,
        "inc": "tags",
    }

    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()

    data = response.json()
    recordings = data.get("recordings", [])

    if not recordings:
        return None

    return recordings[0]


def build_fallback_tags(title, artist_name, genre, mood, era):
    parts = []

    if title:
        parts.append(str(title).strip().lower())

    if artist_name:
        parts.append(str(artist_name).strip().lower())

    if genre:
        parts.append(str(genre).strip().lower())

    if mood:
        parts.append(str(mood).strip().lower())

    if era:
        parts.append(str(era).strip().lower())

    return " ".join(parts).strip()


def map_musicbrainz_to_track(recording):
    track_id = recording.get("id")
    title = recording.get("title", "Unknown")

    artist_name = "Unknown"
    artist_credit = recording.get("artist-credit", [])

    if artist_credit:
        first_artist = artist_credit[0]

        if isinstance(first_artist, dict):
            artist_name = first_artist.get("name", "Unknown")

    year = extract_year(recording.get("first-release-date"))

    tag_names = []

    for tag in recording.get("tags", []):
        name = normalize_tag_name(tag.get("name"))

        if name:
            tag_names.append(name)

    raw_tags = " ".join(tag_names).strip()

    genre = infer_genre_from_tags(raw_tags)
    mood = infer_mood_from_tags(raw_tags)
    era = infer_era(year)

    if raw_tags:
        tags = raw_tags
    else:
        tags = build_fallback_tags(title, artist_name, genre, mood, era)

    tempo_bpm = 120
    popularity = 50

    return {
        "track_id": str(track_id),
        "title": title,
        "artist_name": artist_name,
        "genre": genre,
        "mood": mood,
        "era": era,
        "tempo_bpm": tempo_bpm,
        "year": year,
        "popularity": popularity,
        "tags": tags,
    }


def find_existing_track(title, artist_name):
    normalized_title = normalize_text(title)
    normalized_artist = normalize_text(artist_name)

    for track in Track.objects.all():
        if (
            normalize_text(track.title) == normalized_title and
            normalize_text(track.artist_name) == normalized_artist
        ):
            return track

    return None


def import_demo_tracks():
    imported = 0
    updated_existing = 0
    created_new = 0

    for item in DEMO_QUERIES:
        try:
            print(f"Searching MusicBrainz: {item['title']} - {item['artist']}")

            recording = search_recording(item["title"], item["artist"])

            if not recording:
                print(f"No MusicBrainz result for: {item['title']} - {item['artist']}")
                continue

            mapped = map_musicbrainz_to_track(recording)

            existing_track = find_existing_track(mapped["title"], mapped["artist_name"])

            if existing_track:
                existing_track.genre = mapped["genre"]
                existing_track.mood = mapped["mood"]
                existing_track.era = mapped["era"]
                existing_track.tempo_bpm = mapped["tempo_bpm"]
                existing_track.year = mapped["year"]
                existing_track.popularity = mapped["popularity"]
                existing_track.tags = mapped["tags"]
                existing_track.save()

                updated_existing += 1
                imported += 1

                print(
                    f"Updated existing track: {existing_track.title} - "
                    f"{existing_track.artist_name} ({existing_track.track_id}) | "
                    f"tags: {existing_track.tags}"
                )
            else:
                Track.objects.update_or_create(
                    track_id=mapped["track_id"],
                    defaults=mapped,
                )

                created_new += 1
                imported += 1

                print(
                    f"Imported new track: {mapped['title']} - {mapped['artist_name']} "
                    f"({mapped['track_id']}) | tags: {mapped['tags']}"
                )

            time.sleep(1.1)

        except Exception as e:
            print(f"Failed: {item['title']} - {item['artist']}")
            print(e)

    print(f"Done. Processed {imported} MusicBrainz tracks.")
    print(f"Updated existing tracks: {updated_existing}")
    print(f"Created new tracks: {created_new}")


if __name__ == "__main__":
    import_demo_tracks()