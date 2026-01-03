import urllib.request
import zipfile
import os

# URL for the MobileNet V1 Quantized model (Small & Fast)
url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
file_name = "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

print(f"⬇️ Downloading {file_name}...")
urllib.request.urlretrieve(url, file_name)

print("📦 Unzipping...")
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(".")

print("✅ Done! You should see 'detect.tflite' and 'labelmap.txt' in your folder.")