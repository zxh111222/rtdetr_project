# Create your views here.


from ultralytics.predict import detect_objects
from .forms import DetectionForm
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile
from .utils import get_weather_info
import requests
import cv2
import base64
import os
import csv
from PIL import Image


# CSV文件路径
CSV_FILE_PATH = 'users.csv'

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # 检查用户名、密码和确认密码是否都输入了
        if username and password and confirm_password:
            # 检查用户名是否已经存在
            if not is_user_exists(username):
                # 检查密码和确认密码是否一致
                if password == confirm_password:
                    # 写入CSV文件
                    with open(CSV_FILE_PATH, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([username, password])

                    # 注册成功后重定向到登录页面
                    return redirect('login')
                else:
                    # 密码和确认密码不一致，返回注册页面并显示错误信息
                    return render(request, 'detection/register.html', {'error_message': '两次密码输入不一致'})

            else:
                # 用户名已存在，返回注册页面并显示错误信息
                return render(request, 'detection/register.html', {'error_message': '用户名已存在'})
        else:
            # 输入不完整，返回注册页面并显示错误信息
            return render(request, 'detection/register.html', {'error_message': '请输入完整'})

    return render(request, 'detection/register.html')



def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        cecky = request.POST.get('cecky')  # 获取复选框的值

        # 检查用户名和密码是否为空
        if not username or not password:
            return render(request, 'detection/login.html', {'error_message': '用户名或密码不能为空'})

        # 检查用户名和密码是否匹配，并且检查复选框是否被选中
        if is_valid_credentials(username, password) and cecky == '1':
            # 登录成功后设置session变量，并重定向到首页
            request.session['login_successful'] = True
            return redirect('index')
        else:
            # 如果用户名和密码正确，但是复选框未被选中，输出错误信息
            if is_valid_credentials(username, password) and cecky != '1':
                return render(request, 'detection/login.html', {'error_message': '请勾选我们的协议'})
            # 登录失败，返回登录页面并显示错误信息
            return render(request, 'detection/login.html', {'error_message': '用户名或密码错误'})

    return render(request, 'detection/login.html')




def is_user_exists(username):
    with open(CSV_FILE_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == username:  # 检查行是否非空
                return True
    return False



def is_valid_credentials(username, password):
    # 打开 CSV 文件进行读取
    with open(CSV_FILE_PATH, mode='r', newline='') as file:
        reader = csv.reader(file)

        # 遍历 CSV 文件的每一行
        for row in reader:
            # 检查用户名和密码是否匹配当前行
            if len(row) >= 2 and row[0] == username and row[1] == password:
                return True  # 如果匹配成功，返回 True

    return False  # 如果遍历完所有行都没有匹配成功，返回 False


def logout(request):
    # 清除登录状态
    if 'login_successful' in request.session:
        del request.session['login_successful']

    # 重定向到登录页面
    return redirect('login')

# def index(request):
#     return render(request, 'detection/index.html')
def index(request):
    api_key = "f606ea8c59b723799951f1afa5e8f55a"
    city_name = "Xiamen"

    weather_info = get_weather_info(api_key, city_name)
    print(weather_info)

    return render(request, 'detection/index.html', {'weather_info': weather_info})

@csrf_exempt
def detect_objects_view(request):
    if request.method == 'POST':
        form = DetectionForm(request.POST, request.FILES)
        if form.is_valid():
            pt_file_path = request.FILES['pt_file'].temporary_file_path()  # 获取上传的.pt文件路径
            image_file_path = request.FILES['image_file'].temporary_file_path()  # 获取上传的图片文件路径

            try:
                print(pt_file_path)
                print(image_file_path)
                print("111111111111111111111111111111111111111111111111111111111111")
                # 执行目标检测并获取预测后的图像
                predicted_image = detect_objects(image_file_path, pt_file_path)
                print("2222222222222222222222222222222222222222222222222222222222")

                # 将预测后的图像转换为 Base64 编码的字符串
                _, buffer = cv2.imencode('.jpg', predicted_image)
                predicted_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # 返回预测结果给前端
                return JsonResponse({'image': predicted_image_base64})
            except Exception as e:
                # 如果出现错误，返回错误消息给前端
                return JsonResponse({'error': str(e)}, status=500)
        else:
            # 如果表单验证失败，则返回表单错误信息给前端
            return JsonResponse({'error': form.errors}, status=400)

    # 如果请求方法不是 POST，则返回错误消息给前端
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
























