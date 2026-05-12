from PIL import Image

img = Image.open("wood.jpg").convert("RGB")
img.save("wood_converted.jpg", "JPEG")
