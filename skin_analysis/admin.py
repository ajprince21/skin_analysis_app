from django.contrib import admin
from .models import SkinAnalysis

class SkinAnalysisAdmin(admin.ModelAdmin):
    list_display = ('image', 'skin_tone', 'skin_type', 'percentage', 'acne_types', 'redness_score', 'pores_score', 'pigmentation_score', 'created_at')
    search_fields = ('skin_tone', 'skin_type', 'acne_types') 
    list_filter = ('skin_type', 'created_at') 

admin.site.register(SkinAnalysis, SkinAnalysisAdmin)