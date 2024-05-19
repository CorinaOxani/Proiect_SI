import numpy as np
import cv2
import pickle
# Dictionary for class labels (assuming classes are indexed from 0)
class_labels = {
    0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'End of Speed limit 80 km/h',
    6: 'Speed Limit 100 km/h', 7: 'Speed Limit 120 km/h', 8: 'No passing',
    9: 'No passing for vehicles over 3.5 metric tons', 10: 'Right-of-way at the next intersection',
    11: 'Priority road', 12: 'Yield', 13: 'Stop', 14: 'No vehicles',
    15: 'Vehicles over 3.5 metric tons prohibited', 16: 'No entry',
    17: 'General caution', 18: 'Dangerous curve to the left',
    19: 'Dangerous curve to the right', 20: 'Double curve',
    21: 'Bumpy road', 22: 'Slippery road', 23: 'Road narrows on the right',
    24: 'Road work', 25: 'Traffic signals', 26: 'Pedestrians',
    27: 'Children crossing', 28: 'Bicycles crossing', 29: 'Beware of ice/snow',
    30: 'Wild animals crossing', 31: 'End of all speed and passing limits',
    32: 'Turn right ahead', 33: 'Turn left ahead', 34: 'Ahead only',
    35: 'Go straight or right', 36: 'Go straight or left', 37: 'Keep right',
    38: 'Keep left', 39: 'Roundabout mandatory', 40: 'End of no passing',
    41: 'End no passing vehicle with a weight greater than 3.5 tons'
}
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

False

from tensorflow import keras

# Asigură-te că 'traffic_classifier.h5' este calea corectă la fișierul modelului salvat.
my_model = keras.models.load_model('traffic_classifier.h5')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize and normalize the image
    img = cv2.resize(img, (30, 30))  # Resize to the size model was trained on
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

    
def getClassId(classNo):
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'End of Speed limit 80 km/h'
    elif classNo == 6: return 'Speed Limit 100 km/h'
    elif classNo == 7: return 'Speed Limit 120 km/h'
    elif classNo == 8: return 'No passing'
    elif classNo == 9: return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 10: return 'Right-of-way at the next intersection'
    elif classNo == 11: return 'Priority road'
    elif classNo == 12: return 'Yield'
    elif classNo == 13: return 'Stop'
    elif classNo == 14: return 'No vehicles'
    elif classNo == 15: return 'Veh > 3.5 metric tons prohibited'
    elif classNo == 16: return 'No entry'
    elif classNo == 17: return 'General caution'
    elif classNo == 18: return 'Dangerous curve to the left'
    elif classNo == 19: return 'Dangerous curve to the right'
    elif classNo == 20: return 'Double curve'
    elif classNo == 21: return 'Bumpy road'
    elif classNo == 22: return 'Slippery road'
    elif classNo == 23: return 'Road narrows on the right'
    elif classNo == 24: return 'Road work'
    elif classNo == 25: return 'Traffic signals'
    elif classNo == 26: return 'Pedestrians'
    elif classNo == 27: return 'Children crossing'
    elif classNo == 28: return 'Bicycles crossing'
    elif classNo == 29: return 'Beware of ice/snow'
    elif classNo == 30: return 'Wild animals crossing'
    elif classNo == 31: return 'End of all speed and passing limits'
    elif classNo == 32: return 'Turn right ahead'
    elif classNo == 33: return 'Turn left ahead'
    elif classNo == 34: return 'Ahead only'
    elif classNo == 35: return 'Go straight or right'
    elif classNo == 36: return 'Go straight or left'
    elif classNo == 37: return 'Keep right'
    elif classNo == 38: return 'Keep left'
    elif classNo == 39: return 'Roundabout mandatory'
    elif classNo == 40: return 'End of no passing'
    elif classNo == 41: return 'End no passing vehicle with a weight greater than 3.5 tons'

while True:
    success, imgOriginal = cap.read()
    if not success:
        break  # If the frame is not successfully read, exit the loop

    img = preprocessing(imgOriginal)  # Preprocess the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = my_model.predict(img)  # Predict
    classIndex = np.argmax(predictions, axis=1)[0]  # Get the class index
    probabilityValue = np.max(predictions)  # Get the probability value

    if probabilityValue > 0.60:  # Only display predictions above a threshold
        cv2.putText(imgOriginal, f"Class: {class_labels.get(classIndex, 'Unknown')}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"Probability: {probabilityValue:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
