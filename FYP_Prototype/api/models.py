from django.db import models

class Track(models.Model):
    track_id = models.CharField(max_length=64, unique=True)
    title = models.CharField(max_length=255)
    artist_name = models.CharField(max_length=255)

    genre = models.CharField(max_length=64, blank=True, null=True)
    mood = models.CharField(max_length=64, blank=True, null=True)
    era = models.CharField(max_length=32, blank=True, null=True)

    tempo_bpm = models.FloatField(null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    popularity = models.IntegerField(null=True, blank=True)
    tags = models.TextField(blank=True)
    feature_vector = models.JSONField(default=list)

    def __str__(self):
        return f"{self.track_id} - {self.title}"
