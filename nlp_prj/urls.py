from django.urls import path
from . import views
from .views import article_list,article_detail
urlpatterns = [
    path('api/', views.article_list, name="nlp_view"),
    path('', views.index, name="home_View"),
    path('api/detail/<int:pk>/', views.article_detail, name="sa"),
    path('test/', views.test, name="nlp_view"),
]
