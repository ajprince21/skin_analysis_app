from django.db import models

# Create your models here.

class SkinAnalysis(models.Model):
    image = models.ImageField(upload_to='uploads/')
    acne_score = models.FloatField(null=True, blank=True)
    eye_area_condition = models.FloatField(null=True, blank=True)
    uniformness_score = models.FloatField(null=True, blank=True)
    redness_score = models.FloatField(null=True, blank=True)
    pores_score = models.FloatField(null=True, blank=True)
    lines_score = models.FloatField(null=True, blank=True)
    hydration_score = models.FloatField(null=True, blank=True)
    pigmentation_score = models.FloatField(null=True, blank=True)
    chronological_age = models.IntegerField(null=True, blank=True)
    perceived_age = models.IntegerField(null=True, blank=True)
    skin_tone = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f'Skin Analysis for {self.image.name}'
