import requests
from core.settings import settings

url = settings.server_url_incidents

headers = {
    "accept": "application/json",
    "Authorization": settings.server_api_token,
    "Content-Type": "application/json"
}


def send_incident(type, time, img, location):
    data = {
        "type": type,
        "time": time,
        "img": str(img),
        "location": location
    }

    response = requests.post(url, headers=headers, json=data)

    print(response.json())


# data = {
#     "type": "test",
#     "time": "test",
#     "img": "string"
# }

# response = requests.post(url, headers=headers, json=data)

# print(response.json())