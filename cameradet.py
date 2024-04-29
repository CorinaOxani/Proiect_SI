import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from imutils.video import VideoStream
import time

# Image size for the model
imgSize = 30

# Dictionary to label all traffic sign classes
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing vehicle with a weight greater than 3.5 tons'
}

# Load model
new_model = tf.keras.models.load_model('traffic_classifier.h5')
print(new_model.summary())  # Display the model summary to check structure

# Initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow time for camera to warm up

# Process video frames
try:
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (imgSize, imgSize))
        test_image = image.img_to_array(frame)
        test_image = np.expand_dims(test_image, axis=0)

        # Prediction
        preds = new_model.predict(test_image)
        print("Raw predictions:", preds)  # Display raw prediction values
        predicted_class = np.argmax(preds, axis=1)[0]
        direction = classes.get(predicted_class + 1, 'Unknown')  # Adjust if necessary
        confidence = np.max(preds)

        if confidence < 0.5:  # assuming 0.5 is the confidence threshold
            print("No traffic sign detected.")
        else:
            direction = classes.get(predicted_class + 1, 'Unknown')
        print(f"Detected: {direction} with confidence {confidence}")

        # Display direction
        print(direction)

        # Add prediction text on frame
        #cv2.putText(frame, 'Direction: Test', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display frame
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

finally:
    # Cleanup and close resources
    cv2.destroyAllWindows()
    vs.stop()
