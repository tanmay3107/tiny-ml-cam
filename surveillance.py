import cv2
import numpy as np
import tensorflow as tf
import time

# 1. SETUP: Load the TFLite Model
# We use the TFLite Interpreter which is optimized for CPU inference
model_path = "detect.tflite"
label_path = "labelmap.txt"

print("Loading Quantized Model...")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print("❌ Error: Model files not found!")
    print("Did you run 'setup_model.py' first?")
    print(f"Details: {e}")
    exit()

# Get input and output details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Load labels (e.g., Person, Bicycle, Car)
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 2. START CAMERA
# 0 is usually the default built-in webcam.
# If you have an external USB camera, try changing this to 1.
cap = cv2.VideoCapture(0)

print("\n📷 SMART CAMERA ACTIVE")
print("Press 'q' to quit the video feed.")

# Initialize FPS calculation variables
prev_frame_time = 0
new_frame_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Start timer for inference
    start_time = time.time()

    # 3. PRE-PROCESSING
    # Resize to 300x300 (Model's expected input size)
    image_resized = cv2.resize(frame, (width, height))
    
    # Add batch dimension: [300, 300, 3] -> [1, 300, 300, 3]
    input_data = np.expand_dims(image_resized, axis=0)

    # 4. INFERENCE (The "Thinking" Phase)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 5. RETRIEVE RESULTS
    # Output[0] = Bounding Box locations (ymin, xmin, ymax, xmax)
    # Output[1] = Class Indices (0=Person, etc.)
    # Output[2] = Confidence Scores (0.0 to 1.0)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Stop timer
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # Convert to ms

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # 6. DRAWING & ALERTS
    detected_intruder = False
    
    for i in range(len(scores)):
        # Only show detections with > 50% confidence
        if scores[i] > 0.5:
            # Get bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            h, w, _ = frame.shape
            
            # Convert normalized coordinates (0-1) to pixel values
            left = int(xmin * w)
            top = int(ymin * h)
            right = int(xmax * w)
            bottom = int(ymax * h)
            
            # Get label name
            class_id = int(classes[i])
            # Safety check for label index
            if class_id < len(labels):
                label_name = labels[class_id]
            else:
                label_name = "Unknown"

            # Draw the Bounding Box (Green)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw Label Background (Filled Green Box)
            cv2.rectangle(frame, (left, top - 20), (right, top), (0, 255, 0), -1)
            
            # Put Text Label
            label_text = f"{label_name} {int(scores[i]*100)}%"
            cv2.putText(frame, label_text, (left + 5, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Check if specific object is detected
            if label_name == "person":
                detected_intruder = True

    # 7. DISPLAY METRICS
    # Inference Time (Crucial for Edge AI demo)
    cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # FPS Counter
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Intruder Alert
    if detected_intruder:
        cv2.putText(frame, "WARNING: PERSON DETECTED", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Show the video window
    cv2.imshow('TinyML Edge Cam', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()