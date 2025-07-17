import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import cv2

# Provided robot arm control functions
def armLeft(distance):
    print(f"armLeft distance: {distance}")

def armRight(distance):
    print(f"armRight distance: {distance}")

def armUp(distance):
    print(f"armUp distance: {distance}")

def armDown(distance):
    print(f"armDown distance: {distance}")


def cutterOpen(distance):
    print(f"cutterOpen distance: {distance}")

def cutterClose(distance):
    print(f"cutterClose distance: {distance}")

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

# Variables to store previous positions
prev_wrist_x = None
prev_wrist_y = None
prev_finger_distance = None

# Thresholds for movement detection
movement_threshold = 0.02  # Adjust based on testing
finger_distance_threshold = 0.01  # Adjust based on testing

# Main loop
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,  # Draw on the original BGR frame
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # Draw hand connections
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green landmarks
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Red connections
            )


            # Get wrist position for arm movement
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_wrist_x = wrist.x
            current_wrist_y = wrist.y

            # Get thumb and index finger tips for cutter control
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            current_finger_distance = calculate_distance(thumb_tip, index_tip)

            # Initialize previous positions if not set
            if prev_wrist_x is None:
                prev_wrist_x = current_wrist_x
                prev_wrist_y = current_wrist_y
                prev_finger_distance = current_finger_distance
                continue

            # Calculate movement distances
            delta_x = current_wrist_x - prev_wrist_x
            delta_y = current_wrist_y - prev_wrist_y
            delta_finger_distance = current_finger_distance - prev_finger_distance

            # Arm movement control
            if abs(delta_x) > movement_threshold:
                distance = abs(delta_x)
                if delta_x > 0:
                    armRight(distance)
                else:
                    armLeft(distance)

            if abs(delta_y) > movement_threshold:
                distance = abs(delta_y)
                if delta_y > 0:
                    armDown(distance)
                else:
                    armUp(distance)

            # Cutter control
            if abs(delta_finger_distance) > finger_distance_threshold:
                distance = abs(delta_finger_distance)
                if delta_finger_distance > 0:
                    cutterOpen(distance)
                else:
                    cutterClose(distance)

            # Update previous positions
            prev_wrist_x = current_wrist_x
            prev_wrist_y = current_wrist_y
            prev_finger_distance = current_finger_distance

    cv2.imshow('Robot Arm Control with Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()