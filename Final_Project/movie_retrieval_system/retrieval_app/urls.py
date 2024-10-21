from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('movie/<str:filename>/', views.movie_detail, name='movie_detail'),
]