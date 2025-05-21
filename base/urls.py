from django.urls import path
from .views import predictImage, HomePage, SignupPage, LoginPage, LogoutPage, index1,index2,result,index,chatbot_view

urlpatterns = [
    path('', HomePage, name='home'),
    path('index/', index, name='index'),
    path('index1/', index1, name='index1'),
    path('index2/',index2,name='index2'),
    path('signup/', SignupPage, name='signup'),
    path('login/', LoginPage, name='login'),
    path('logout/', LogoutPage, name='logout'),
    path('result/', result, name='result'),
    path('chatbot', chatbot_view, name='chatbot_view'),
    path('predictImage', predictImage, name='predictImage'),
]