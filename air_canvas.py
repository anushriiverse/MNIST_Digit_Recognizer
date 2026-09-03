import cv2 as cv
import numpy as np
import mediapipe as mp
from src.data_preprocess import format_for_model
from src.inference import load_model, predict_digit

MODEL_PATH = "models/mnist_rf_model.pkl"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    exit()

# Initializing mediapipe drawing hand modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils

# Setting up the webcam
cap = cv.VideoCapture(0)
# creating a blank canvas to draw on
canvas = np.zeros((480, 640), dtype=np.uint8)

prev_x, prev_y = 0, 0

print("Index finger only: Draw")
print("Index + Middle Finger: Hover/Move")
print("Press 'p' : Predict the drawn number")
print("Press 'c' : Clear the canvas")
print("Press 'q' : Quit")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the BGR image to RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # if hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip and middle finger tip
            index_tip_y = hand_landmarks.landmark[8].y
            index_joint_y = hand_landmarks.landmark[6].y
            middle_tip_y = hand_landmarks.landmark[12].y
            middle_joint_y = hand_landmarks.landmark[10].y

            # convert normalized coordinates to pixel values
            curr_x = int(hand_landmarks.landmark[8].x * w)
            curr_y = int(hand_landmarks.landmark[8].y * h)

            # Check if the index finger is up and middle finger is down
            index_up = index_tip_y < index_joint_y
            middle_up = middle_tip_y < middle_joint_y

            # Function 1: hover when both fingers are up
            if index_up and middle_up:
                prev_x, prev_y = 0,0
                cv.circle(frame, (curr_x, curr_y), 15, (255,0,255), cv.FILLED)

            # Function 2: draw when only index finger is up
            elif index_up and not middle_up:
                cv.circle(frame, (curr_x, curr_y), 15, (0,255,0), cv.FILLED)

                # If this is the first point, set it as the previous point
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = curr_x, curr_y 

                # draw thick line on the canvas
                cv.line(canvas, (prev_x, prev_y), (curr_x, curr_y), 255, 15)
                prev_x, prev_y = curr_x, curr_y

    # Display the canvas and the frame
    canvas_bgr = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
    frame = cv.addWeighted(frame, 1, canvas_bgr, 1, 0)

    cv.imshow("Air Canvas", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros((480, 640), dtype=np.uint8)
        print("Canvas cleared.")
    elif key == ord('p'):
        inverted_canvas = cv.bitwise_not(canvas)
        try:
            processed_image = format_for_model(inverted_canvas)
            prediction = predict_digit(model, processed_image)
            print(f"Predicted digit: {prediction}")
        except Exception as e:
            print(f"Error during prediction: {e}")
    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
