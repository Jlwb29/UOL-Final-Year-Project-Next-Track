from rest_framework import serializers
from .models import Track


class TrackSerializer(serializers.ModelSerializer):
    def validate_track_id(self, value):
        if not value or not str(value).strip():
            raise serializers.ValidationError("track_id cannot be empty")
        return str(value).strip()

    def validate_title(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("title cannot be empty")
        return value.strip()

    def validate_artist_name(self, value):
        if not value or not value.strip():
            raise serializers.ValidationError("artist_name cannot be empty")
        return value.strip()

    def validate_tempo_bpm(self, value):
        if value is None:
            return value

        if value < 40 or value > 250:
            raise serializers.ValidationError("tempo_bpm must be between 40 and 250")
        return value

    def validate_year(self, value):
        if value is None:
            return value

        if value < 1900 or value > 2026:
            raise serializers.ValidationError("year must be between 1900 and 2026")
        return value

    def validate_genre(self, value):
        if value is None:
            return value

        value = value.strip().lower()

        allowed = [
            "pop", "rock", "hiphop", "rap", "edm",
            "jazz", "classical", "rnb", "metal", "indie"
        ]

        if value not in allowed:
            raise serializers.ValidationError(f"genre must be one of {allowed}")

        return value

    def validate_mood(self, value):
        if value is None:
            return value

        value = value.strip().lower()

        allowed = [
            "happy", "sad", "relaxed", "energetic",
            "chill", "angry", "romantic", "focus"
        ]

        if value not in allowed:
            raise serializers.ValidationError(f"mood must be one of {allowed}")

        return value

    def validate_tags(self, value):
        if not value:
            return ""

        tags = str(value).lower().split(",")

        cleaned = []
        seen = set()

        for t in tags:
            t = t.strip()

            if not t:
                continue

            if t not in seen:
                cleaned.append(t)
                seen.add(t)

        return ",".join(cleaned)

    class Meta:
        model = Track
        fields = [
            "track_id",
            "title",
            "artist_name",
            "genre",
            "mood",
            "era",
            "tempo_bpm",
            "year",
            "popularity",
            "tags",
        ]


class TrackReadSerializer(serializers.ModelSerializer):
    title = serializers.SerializerMethodField()
    artist_name = serializers.SerializerMethodField()
    genre = serializers.SerializerMethodField()
    mood = serializers.SerializerMethodField()

    def format_text(self, value):
        if not value:
            return value
        return str(value).strip().title()

    def get_title(self, obj):
        return self.format_text(obj.title)

    def get_artist_name(self, obj):
        return self.format_text(obj.artist_name)

    def get_genre(self, obj):
        return self.format_text(obj.genre)

    def get_mood(self, obj):
        return self.format_text(obj.mood)

    class Meta:
        model = Track
        fields = [
            "track_id",
            "title",
            "artist_name",
            "genre",
            "mood",
            "era",
            "tempo_bpm",
            "year",
            "popularity",
            "tags",
        ]


class RecommendQuerySerializer(serializers.Serializer):
    seed_ids = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_empty=True,
        default=list
    )
    genre = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    mood = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    era = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    tempo_bpm = serializers.FloatField(required=False, allow_null=True)
    year = serializers.IntegerField(required=False, allow_null=True)
    tags = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    limit = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=50,
        default=10
    )
    transparent = serializers.BooleanField(
        required=False,
        default=False
    )

    def validate_genre(self, value):
        if value in [None, ""]:
            return None

        value = value.strip().lower()
        allowed = [
            "pop", "rock", "hiphop", "rap", "edm",
            "jazz", "classical", "rnb", "metal", "indie"
        ]

        if value not in allowed:
            raise serializers.ValidationError(f"genre must be one of {allowed}")

        return value

    def validate_mood(self, value):
        if value in [None, ""]:
            return None

        value = value.strip().lower()
        allowed = [
            "happy", "sad", "relaxed", "energetic",
            "chill", "angry", "romantic", "focus"
        ]

        if value not in allowed:
            raise serializers.ValidationError(f"mood must be one of {allowed}")

        return value

    def validate_tempo_bpm(self, value):
        if value is None:
            return value

        if value < 40 or value > 250:
            raise serializers.ValidationError("tempo_bpm must be between 40 and 250")
        return value

    def validate_year(self, value):
        if value is None:
            return value

        if value < 1900 or value > 2026:
            raise serializers.ValidationError("year must be between 1900 and 2026")
        return value

    def validate_tags(self, value):
        if not value:
            return ""

        tags = str(value).lower().split(",")

        cleaned = []
        seen = set()

        for t in tags:
            t = t.strip()

            if not t:
                continue

            if t not in seen:
                cleaned.append(t)
                seen.add(t)

        return ",".join(cleaned)

    def validate(self, attrs):
        seed_ids = attrs.get("seed_ids", [])
        genre = attrs.get("genre")
        mood = attrs.get("mood")
        era = attrs.get("era")
        tempo_bpm = attrs.get("tempo_bpm")
        year = attrs.get("year")
        tags = attrs.get("tags")

        has_preferences = any([
            genre,
            mood,
            era,
            tempo_bpm is not None,
            year is not None,
            tags,
        ])

        if not seed_ids and not has_preferences:
            raise serializers.ValidationError(
                "Provide at least one seed_id or one preference field."
            )

        return attrs