import requests

request_template = {
        "messages": [{"role": "user", "content": "Hello"}],
        "language": "English"
    }

response = requests.post("http://localhost:9000/chat", json=request_template)

if response.status_code == 200:
    print("Request successful!")
    
    for chunk in response.iter_content(1024):
        print(chunk)
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)
