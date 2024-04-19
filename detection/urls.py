from django.urls import path
from . import views


urlpatterns = [
    path('', views.login, name='login'),  # 默认路径指向登录页面
    path('login/', views.login, name='login'),  # 登录页面路径
    path('register/', views.register, name='register'),  # 添加注册页面路径
    path('return_to_login/', views.login, name='return_to_login'),
    path('logout/', views.logout, name='logout'),
    path('index/', views.index, name='index'),
    # path('index_other/', views.index_other, name='index_other'),
    path('detect_objects_view/', views.detect_objects_view, name='detect_objects_view'),
]
