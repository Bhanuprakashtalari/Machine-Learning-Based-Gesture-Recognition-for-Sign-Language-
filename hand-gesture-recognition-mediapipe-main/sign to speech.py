# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Define the hand signal classes and labels
class_names = ['Fist', 'Open hand', 'Thumbs up', 'Peace sign']
num_classes = len(class_names)

# Set the time interval for speech output (in seconds)
speech_interval = 3
last_speech_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand landmarks as a flat array
            flattened_landmarks = np.array([landmark.x for landmark in hand_landmarks.landmark] +
                                            [landmark.y for landmark in hand_landmarks.landmark])

            # Simulated classifier (replace with your own classifier)
            # In this example, classify based on the position of the middle finger tip
            middle_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x

            # Classify hand signal
            if middle_finger_tip_x < 0.3:
                label = 0 # Fist
            elif middle_finger_tip_x > 0.7:
                label = 2 # Thumbs up
            else:
                label = 1 # Open hand

            # Display the predicted hand signal
            prediction_text = f'Prediction: {class_names[label]}'
            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if it's time for speech output
            current_time = time.time()
            if current_time - last_speech_time >= speech_interval:
                # Text-to-speech
                tts = gTTS(text=prediction_text, lang='en')
                tts.save("output.mp3")
                os.system("start output.mp3")
                # Update the last speech time
                last_speech_time = current_time

    cv2.imshow('Hand Signal Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
