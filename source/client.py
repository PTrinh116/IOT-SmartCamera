import requests
import base64


# Function to encode image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

url = 'http://127.0.0.1:8000/recog'

image_path = r"D:\HKVI\Iot\IotTcH\lab5\source\Dataset\processed\Trinh\photo_5.png"
encoded_image_data = encode_image_to_base64(image_path)


# Parameters to send in the POST request
data = {
    'image': encoded_image_data,
    'w': 100,  #  width
    'h': 100   #  height
}

# Sending the POST request
response = requests.post(url, data=data)

# Printing the response
print("Response from server:")
print(response.text)
