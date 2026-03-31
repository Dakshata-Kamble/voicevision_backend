import requests

url = "http://10.186.2.22:5000/read-text"

files = {"image": open("test.jpg", "rb")}

res = requests.post(url, files=files)

print(res.json())