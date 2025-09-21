import base64
# Replace with your image filename
image_path = "oldman_image.jpg"
# Open and encode the image
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
print(encoded_string)
