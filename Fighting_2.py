import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pynput.keyboard import Controller, Key

keyboard = Controller()

kick_lock = False
kick_lock_time = 0
kick_cooldown = 0.5

punch_lock = False
punch_lock_time = 0
punch_cooldown = 0.5

special_lock = False
special_lock_time = 0
special_cooldown = 1.0

prev_lw = None
prev_lw_time = None


def find_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


class poseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


last_position = {'x': None, 'y': None}
wasd_pressed = set()

def press_key_once(key):
    keyboard.press(key)
    keyboard.release(key)

def detect_actions_and_trigger_keys(lmList):
    global last_position, wasd_pressed
    global kick_lock, kick_lock_time, kick_cooldown
    global punch_lock, punch_lock_time, punch_cooldown
    global special_lock, special_lock_time, special_cooldown
    global prev_lw, prev_lw_time

    if not lmList or len(lmList) < 33:
        return

    current_time = time.time()

    head = lmList[0]
    rw, re = lmList[16], lmList[14]
    lw, le = lmList[15], lmList[13]
    ls = lmList[11]  # Left shoulder
    ra = lmList[28]
    rk = lmList[26]
    rh = lmList[24]
    lh = lmList[23]

    # Punch detection
    if prev_lw and prev_lw_time:
        dx = lw[1] - prev_lw[0]
        dt = current_time - prev_lw_time
        velocity = dx / dt if dt > 0 else 0
        elbow_angle = find_angle([ls[1], ls[2]], [le[1], le[2]], [lw[1], lw[2]])

        if not punch_lock and velocity < -300 and elbow_angle > 160:
            press_key_once('h')
            print(f"Punch! v={int(velocity)} angle={int(elbow_angle)}")
            punch_lock = True
            punch_lock_time = current_time

    if punch_lock and current_time - punch_lock_time > punch_cooldown:
        punch_lock = False

    prev_lw = (lw[1], lw[2])
    prev_lw_time = current_time

    # Kick detection
    hip = rh[1:]
    knee = rk[1:]
    ankle = ra[1:]
    knee_angle = find_angle(hip, knee, ankle)

    if knee_angle < 130 and not kick_lock:
        press_key_once('g')
        print(f"Kick! angle={int(knee_angle)}")
        kick_lock = True
        kick_lock_time = current_time
    elif kick_lock and current_time - kick_lock_time > kick_cooldown:
        kick_lock = False

    # Special detection (arms wide apart)
    hand_distance = abs(rw[1] - lw[1])
    if hand_distance > 500 and not special_lock:
        press_key_once('j')
        print("Special move!")
        special_lock = True
        special_lock_time = current_time
    elif special_lock and current_time - special_lock_time > special_cooldown:
        special_lock = False

    # Block detection
    if rw[2] < head[2] and lw[2] < head[2]:
        press_key_once(Key.space)
        print("Block")

    # Movement detection
    mid_x = (lh[1] + rh[1]) // 2
    mid_y = (lh[2] + rh[2]) // 2
    new_keys = set()

    if last_position['x'] is not None:
        dx = mid_x - last_position['x']
        dy = mid_y - last_position['y']

        if dx > 10:
            keyboard.press('d')
            new_keys.add('d')
        elif dx < -10:
            keyboard.press('a')
            new_keys.add('a')

        if dy > 10:
            keyboard.press('s')
            new_keys.add('s')
        elif dy < -10:
            keyboard.press('w')
            new_keys.add('w')

    for key in wasd_pressed - new_keys:
        keyboard.release(key)

    wasd_pressed = new_keys
    last_position['x'] = mid_x
    last_position['y'] = mid_y

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            detect_actions_and_trigger_keys(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Pose Fighter Controller", img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
