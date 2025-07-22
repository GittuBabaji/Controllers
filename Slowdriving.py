import cv2
import mediapipe as mp
import math
import numpy as np
import keyboard

class HandSteering:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.steering = 0
        self.prev_steering = 0
        self.steering_smooth_factor = 0.2

        self.throttle = 0
        self.brake = 0

        self.steering_deadzone = 0.2
        self.throttle_threshold = 0.3
        self.brake_threshold = 0.18

        self.max_angle = 45

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

            wrist = [landmarks[0].x, landmarks[0].y]
            middle_base = [landmarks[9].x, landmarks[9].y]
            dx = middle_base[0] - wrist[0]
            dy = middle_base[1] - wrist[1]
            angle = math.degrees(math.atan2(dy, dx))

            steering_norm = np.clip(angle / self.max_angle, -1, 1)
            self.steering = (
                self.prev_steering * (1 - self.steering_smooth_factor)
                + steering_norm * self.steering_smooth_factor
            )
            self.prev_steering = self.steering

            thumb_tip = [landmarks[4].x, landmarks[4].y]
            pinky_tip = [landmarks[20].x, landmarks[20].y]
            palm_width = math.dist(thumb_tip, pinky_tip)

            if palm_width > self.throttle_threshold:
                self.throttle = int(np.interp(palm_width, [self.throttle_threshold, 1], [0, 100]))
                self.brake = 0
            elif palm_width < self.brake_threshold:
                self.brake = int(np.interp(palm_width, [0, self.brake_threshold], [100, 0]))
                self.throttle = 0
            else:
                self.throttle = 0
                self.brake = 0

            self.send_commands()
            frame = self.draw_hud(frame, landmarks)

        return frame

    def send_commands(self):
        keyboard.release('a')
        keyboard.release('d')
        keyboard.release('w')
        keyboard.release('s')

        if self.steering < -self.steering_deadzone:
            keyboard.press('a')
        elif self.steering > self.steering_deadzone:
            keyboard.press('d')

        if self.throttle > 0:
            keyboard.press('w')
        elif self.brake > 0:
            keyboard.press('s')

    def draw_hud(self, frame, landmarks):
        h, w = frame.shape[:2]

        cv2.circle(frame, (w // 2, h // 2), 100, (100, 100, 100), 2)
        angle_rad = self.steering * math.radians(self.max_angle)
        end_x = int(w // 2 + 90 * math.sin(angle_rad))
        end_y = int(h // 2 - 90 * math.cos(angle_rad))
        cv2.line(frame, (w // 2, h // 2), (end_x, end_y), (0, 255, 0), 3)

        cv2.rectangle(frame, (50, h - 150), (100, h - 50), (100, 100, 100), 2)
        cv2.rectangle(frame, (50, h - 150), (100, h - 50 - int(self.throttle)), (0, 255, 0), -1)

        cv2.rectangle(frame, (150, h - 150), (200, h - 50), (100, 100, 100), 2)
        cv2.rectangle(frame, (150, h - 50), (200, h - 50 + int(self.brake)), (0, 0, 255), -1)

        cv2.putText(frame, f"Steering: {int(self.steering * 100)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

def main():
    cap = cv2.VideoCapture(0)
    controller = HandSteering()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = controller.process_frame(frame)
        cv2.imshow('Hand Steering', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    keyboard.release('a')
    keyboard.release('d')
    keyboard.release('w')
    keyboard.release('s')

if __name__ == "__main__":
    main()
