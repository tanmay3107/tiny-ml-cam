🕵️ TinyML Smart Surveillance: Real-Time Edge AI

Overview

TinyML Smart Surveillance is a privacy-first, offline intruder detection system designed to run on low-power edge devices (like standard laptops or Raspberry Pi) without needing a GPU or Internet connection.
Unlike cloud-based AI which suffers from latency and privacy concerns, this project uses a Quantized MobileNet SSD model to perform inference locally. It achieves real-time performance (25+ FPS) on a standard CPU, demonstrating how AI can be deployed in resource-constrained IoT environments.

✨ Key Features

Edge-Native Inference: Runs 100% offline using the CPU, ensuring zero latency and maximum privacy.
Model Quantization: Utilizes a int8 quantized TFLite model, reducing model size by 4x while maintaining detection accuracy.
Real-Time Detection: Capable of detecting 90+ object classes (Person, Vehicle, Animal) in <50ms per frame.
Efficiency Metrics: Displays live Inference Time (ms) and FPS to benchmark hardware performance.

🛠️ Tech Stack

Framework: TensorFlow Lite (TFLite) Interpreter
Vision Library: OpenCV
Model: MobileNet V1 SSD (Quantized COCO)
Hardware: Compatible with x86 CPUs, Raspberry Pi, Jetson Nano

🚀 How to Run

1. Install Dependencies
pip install opencv-python tensorflow numpy


2. Setup Model
Run the setup script to download the specific quantized model needed for edge inference:
python setup_model.py

this downloads detect.tflite and labelmap.txt.

3. Start Surveillance
Run the main script to start the camera feed and detection loop:
python surveillance.py
press q to quit the camera feed.

📸 Performance Benchmarks

Metric                                                  Value (Tested on CPU)

Model Size                                              ~4 MB (vs 100MB+ for ResNet)

Inference Time                                          30ms - 50ms

FPS                                                     25 - 35 FPS

Accuracy                                                High confidence (>85%) for "Person" detection

Note: This project demonstrates the viability of running complex computer vision tasks on edge hardware without reliance on cloud compute.