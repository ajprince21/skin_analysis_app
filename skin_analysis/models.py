from django.db import models

class SkinAnalysis(models.Model):
    image = models.ImageField(upload_to='uploads/') 
    skin_tone = models.CharField(max_length=50)
    skin_type = models.CharField(max_length=50)
    percentage = models.CharField(max_length=10)  
    acne_types = models.CharField(max_length=50)  
    redness_score = models.CharField(max_length=10) 
    pores_score = models.CharField(max_length=10)  
    pigmentation_score = models.CharField(max_length=10) 
    created_at = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"Analysis for {self.image.name}"