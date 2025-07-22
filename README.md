# Controllers

Controllers is a Python project that enables you to control browser-based games using hand gestures, powered by [MediaPipe](https://mediapipe.dev/) and [OpenCV](https://opencv.org/). This project is tailored for three popular games:

- [Slow Roads](https://slowroads.io/)
- [Street Fighter Simulator](https://www.crazygames.com/game/street-fighter-simulator)
- [Stickman Fighting 3D](https://www.crazygames.com/game/stickman-fighting-3d)

## Features

- **Hand Gesture Recognition:** Uses MediaPipe to detect and interpret hand gestures in real-time.
- **OpenCV Integration:** Captures webcam input and processes video streams for gesture recognition.
- **Game Control:** Maps recognized hand gestures to keyboard or mouse actions to control the games seamlessly.
- **Plug-and-play:** Can be adapted for other browser-based games with customizable gesture mappings.

## How It Works

1. **Webcam Capture:** OpenCV is used to capture the live video feed from your webcam.
2. **Gesture Detection:** MediaPipe processes the video to track hand landmarks and recognize specific gestures.
3. **Action Mapping:** Detected gestures are mapped to in-game actions, emulating key presses or mouse movements.
4. **Game Interaction:** The mapped actions are sent to the browser window running the target game.

## Supported Games & Example Mappings

| Game                        | Example Gestures           | Mapped Actions                  |
|-----------------------------|---------------------------|---------------------------------|
| Slow Roads                   | Open palm, fist, swipe    | Accelerate, brake, turn         |
| Street Fighter Simulator     | Punch, kick, block poses  | Attack, defend, special moves   |
| Stickman Fighting 3D         | Directional gestures      | Move, punch, kick, jump         |

*You can customize gestures and mappings in the code.*

## Installation

6. **Clone the Repository**
   ```bash
   git clone https://github.com/GittuBabaji/Controllers.git
   cd Controllers
   ```
9. **Run the Controller**
   ```bash
   python controller.py
   ```

## Usage

1. Launch the script.
2. Select the game you want to control.
3. Allow webcam access.
4. Open the game in your browser:
   - [Slow Roads](https://slowroads.io/)
   - [Street Fighter Simulator](https://www.crazygames.com/game/street-fighter-simulator)
   - [Stickman Fighting 3D](https://www.crazygames.com/game/stickman-fighting-3d)
5. Use hand gestures to play!

## Customization

- Edit gesture mappings in `controller.py` to adapt to new games or change controls.
- Adjust sensitivity and gesture recognition parameters as needed.

## Requirements

- Python 3.7+
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/) (for simulating keyboard/mouse events)

Install all requirements with:
```bash
pip install mediapipe opencv-python pyautogui
```
