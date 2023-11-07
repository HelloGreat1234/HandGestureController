import cv2
import pyautogui
import mediapipe as mp
import numpy as np
import threading

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
screenWidth, screenHeight = pyautogui.size()

left_click_pressed = False


scrolling = False

value = 0

prev_x, prev_y = None, None

def condition_satisfied(dist):
    if dist < 0.05:
        return True  
    else :
        return False

def check_condition_wrapper(dist):
    return lambda: check_condition(dist)

def check_condition(dist):
    global value
    if condition_satisfied(dist):  
        value += 1
    else:
        value = 0
        return

    duration = 1  
    threading.Timer(duration, check_condition_wrapper(dist)).start()

def moveMouse(wrist_x,wrist_y):
    global prev_x,prev_y

    mousePositionX = int((screenWidth) * wrist_x)
    mousePositionY = int(screenHeight * wrist_y)

    point1 = (thumb_x, thumb_y)
    point2 = (index_x, index_y)
    dist = np.linalg.norm(np.array(point1) - np.array(point2))

    if dist < 0.05:
        if prev_x is not None and prev_y is not None:
            diff_x, diff_y = mousePositionX - prev_x, mousePositionY - prev_y
            # Do something with the differences, e.g., move the mouse
            pyautogui.moveRel(diff_x*2, diff_y*2, duration=0.1)

    prev_x, prev_y = mousePositionX, mousePositionY


def left_click(distance):
    global value

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'LEFT CLICK',(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)
    
    pyautogui.leftClick()
    print("LEFT CLICK")
    check_condition(distance)
    if value > 6:
        pyautogui.doubleClick()
        value = 0

    cv2.waitKey(1)


while True:
    ret, frame = cap.read()
    frameHeight, frameWidth, _ = frame.shape
    if not ret:
        break

    # frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            wrist_x = hand_landmarks.landmark[9].x
            wrist_y = hand_landmarks.landmark[9].y

            

            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[5].x:
                hand_id = 0  # Left hand
            else:
                hand_id = 1  # Right hand

                        

            # Performing tasks based on the hand ID
            if hand_id == 1:
                
                if thumb_y < middle_y:
                    hand_right_gesture = 'pointing up'
                    if not scrolling:
                        scrolling = True
                        scroll_start = pyautogui.position()
                elif thumb_y > middle_y:
                    hand_right_gesture = 'pointing down'
                    if scrolling:
                        scrolling = False
                        scroll_start = None
                        pyautogui.scroll(-700)
                    else:
                        hand_right_gesture = "other"

                if scrolling:
                    current_pos = pyautogui.position()
                    pyautogui.moveTo(current_pos[0], current_pos[1] + (pyautogui.position()[1] - scroll_start[1]))

                moveMouse(wrist_x,wrist_y)


            elif hand_id == 0:
                # Task for the right hand
                point1 = (thumb_x, thumb_y)
                point2 = (index_x, index_y)
                dist = np.linalg.norm(np.array(point1) - np.array(point2))

                if thumb_y < middle_y:
                    hand_gesture = 'pointing up'
                    if not scrolling:
                        scrolling = True
                        scroll_start = pyautogui.position()
                elif thumb_y > middle_y:
                    hand_gesture = 'pointing down'
                    if scrolling:
                        scrolling = False
                        scroll_start = None
                        pyautogui.scroll(700)
                    else:
                        hand_gesture = "other"

                if scrolling:
                    current_pos = pyautogui.position()
                    pyautogui.moveTo(current_pos[0], current_pos[1] + (pyautogui.position()[1] - scroll_start[1]))

                if dist < 0.05:  
                    left_click(dist)
                

        cv2.imshow("Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()