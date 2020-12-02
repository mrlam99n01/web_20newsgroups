from django.db import models
COLOR_CHOICES = (
    ('Naive','NAIVE'),
    ('SVM', 'SVM'),
    ('Grid','GRID'),
    ('Tree','TREE'),
    ('Linear','LINEAR'),
    ('Kneast','KNEARST'),
    ('Forest','Forsest'),
    ('DL2','Deeplearning_Lam'),
    ('More_1','MORE_1'),
    ('More_2','MORE_2'),
)
class Article(models.Model):
    model_name = models.CharField(max_length=6, choices=COLOR_CHOICES, default='Naive')
    field_name = models.TextField(default ="This is a default value")
    top_1_accuracy = models.CharField(max_length=100, default="top")
    top_1_percentage = models.CharField(max_length=100, default="per1")
    top_2_accuracy=models.CharField(max_length=100,default="top2")
    top_2_percentage = models.CharField(max_length=100, default="per2")
    top_3_accuracy=models.CharField(max_length=100,default="top3")
    top_3_percentage = models.CharField(max_length=100, default="per3")
    top_4_accuracy=models.CharField(max_length=100,default="top4")
    top_4_percentage = models.CharField(max_length=100, default="per4")

    def __str__(self):
        return self.model_name

# Create your models here.
