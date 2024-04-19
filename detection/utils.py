import requests


def get_weather_info(api_key, city_name):
    print("333333333333333333333333333333333333333333333")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric&lang=zh_cn"
    response = requests.get(url)
    data = response.json()


    if response.status_code == 200:
        weather_info = {
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"]["deg"],
            "city_name": data["name"]
        }
        return weather_info
    else:
        return None