import requests

url = "http://127.0.0.1:5000/detect-currency"

files = {"image": open("sample.jpg", "rb")}

response = requests.post(url, files=files)

print(response.json())