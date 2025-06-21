import requests
import json

scoring_uri = "http://a279f33a-856d-4667-b0ba-235ed9f9d5cc.westus2.azurecontainer.io/score"

# Your primary key (replace with actual key you retrieved)
key = "tY8znkPBiOzRffDHJff5R3a4kldhCi7I"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {key}'
}

data = {
    "data": [
        [5.1, 3.5, 1.4, 0.2]
    ]
}

input_data = json.dumps(data)

response = requests.post(scoring_uri, data=input_data, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.json())
