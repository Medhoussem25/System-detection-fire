# YOLO Fire Detection System

## Description
This application leverages the YOLOv8 object detection model to identify instances of fire in video files. The application is built using Python and provides a user-friendly graphical interface created with Tkinter. It includes functionality for:
- Selecting a video file for detection.
- Automatically loading the YOLO model.
- Running the detection process and raising an alert when fire is detected.
- Displaying developer information.

## Features
1. **Video Input**: Load video files in formats such as `.mp4`, `.avi`, and `.mkv`.
2. **Fire Detection**: Detect instances of fire with a predefined confidence level.
3. **Alert Mechanism**: Emit a sound alert (`Beep`) when fire is detected.
4. **Graphical User Interface (GUI)**: User-friendly interface with buttons for loading videos, running detection, and accessing developer information.
5. **Developer Info**: View details about the developer within the application.

## Prerequisites
- Python 3.8 or later
- Required libraries:
  - `ultralytics`
  - `cv2` (OpenCV)
  - `tkinter`
  - `Pillow`
  - `winsound` (Windows-specific)
- A YOLO model file (`fire.pt`) placed in the same directory as the script.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required Python libraries:
   ```bash
   pip install ultralytics opencv-python pillow
   ```

## Usage
1. Ensure the `fire.pt` model file is in the same directory as the script.
2. Run the script:
   ```bash
   python <script-name>.py
   ```
3. Use the graphical interface to:
   - Load a video by clicking **"Charger une vidéo"**.
   - Start detection by clicking **"Lancer la détection"**.
   - View developer information by clicking **"Info Développeur"**.

## Application Interface
- **Title**: "Système Détection Fire"
- **Buttons**:
  - "Charger une vidéo": Load a video file for detection.
  - "Lancer la détection": Start the detection process.
  - "Info Développeur": Display developer details.
- **Footer**: Instruction to quit the detection process by pressing 'q'.

## File Structure
- `fire.pt`: YOLO model file for fire detection.
- `fire-alarm.png`: Icon image for the application.
- `main.py`: The main script to run the application.

## Developer Information
- **Name**: Houssam Bouagal
- **Email**: mouhamedhoussem813@gmail.com

## Notes
- Ensure that the video file and model file paths are accessible.
- Press `q` to stop the detection process while viewing the video.
- Modify the class name `'fire'` in the code to match the fire detection class in your YOLO model if necessary.

## Future Enhancements
- Support for live video feeds.
- Customizable alert mechanisms.
- Integration with external notification systems.

