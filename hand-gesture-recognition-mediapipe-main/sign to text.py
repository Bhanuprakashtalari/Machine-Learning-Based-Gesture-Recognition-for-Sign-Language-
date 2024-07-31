import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Define the hand signal classes and labels
class_names = ['Fist', 'Open hand', 'Thumbs up', 'Peace sign']
num_classes = len(class_names)

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
                label = 0  # Fist
            elif middle_finger_tip_x > 0.7:
                label = 2  # Thumbs up
            else:
                label = 1  # Open hand

            # Display the predicted hand signal
            cv2.putText(frame, f'Prediction: {class_names[label]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Signal Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
