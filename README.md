# rpi5-security-camera

# Setup

### Hardware 
1. Core: Raspberry Pi 5 (8Gb Ram)
2. Storage: Raspberry Pi NVMe M.2 SSD (512Gb)
3. Storage Interface: ACASIS NVMe to USD Adapter (10 Gbps)
4. AI Acceleration: Raspberry Pi AI Hat+ (26 Tops, Hailo 8L Accelerator)
5. Vision Input: Raspberry Pi Camera Module 3

### Additional Information
* Operating System: Raspberry Pi OS
* Python Installation: `\usr\bin\python3`

# Libraries

## (1) Camera Control Layer: `picamera2`

Description: This library is built specifically for Raspberry Pi hardware to interface with the camera system. It can start/stop the camera, capture images or video frames, and control camera settings such as resolution, frame rate, exposure, and more. Thus, picamera2 is a Python interface that sits on top of libcamera, which manages communication between the application layer and the camera hardware. In short picamera2 = python api, libcamera = system camera pipeline, camera drivers = hardware interface, camera hardware = image sensor. 

Example:
```
from picamera2 import Picamera2
# Create a camera object that connects to the camera hardware, prepares the camera pipeline via libcamera, and allocates resources
camera = Picamera2()
# Start the camera stream so the camera begins capturing frames continuously, sending them into a buffer              
camera.start()
# Retreive the latest frame from the camera buffer and convert it into a numpy array
frame = camera.capture_array()   
```

Installation Notes:
* To check if picamera2 is installed: `python -c "from picamera2 import Picamera2; print('ok')"`
* To find where python is importing it from: `python -c "import picamera2; print(picamera2.__file__)"`
* In most cases on Raspberry Pi OS, picamera2 is installed as a system package via APT, not pip. If installed, it is typically located in: `/usr/lib/python3/dist-packages/`
* If it is not installed, you can install it with: `sudo apt install python3-picamera2`. This installs the Python module along with required system integrations for libcamera support.

Additional Notes:
* Testing the camera system: To verify the camera hardware and driver stack is working, run `libcamera-hello`. This Initializes the camera system (loads drivers and configures libcamera), Connects to the Raspberry Pi Camera Module 3, Starts a temporary video stream, and Displays a live preview window
* Using picamera2 with a venv: picamera2 is a system level hardware library, not a normal pure-python package, meaning, it depends on libcamera, system camera drivers, and os-levl shared libraries. Thats why its normally installed with apt. When you create a venv, you get an isolated python environment with seperate site packages, but you still have access to system libraries (optionally). However, a venv does not automatically include system packages like `/usr/lib/python3/dist-packages/`. So if you are using a venv you can simply create a venv with system packages enabled (i.e. run `python -m venv myenv --system-site-packages`). This allows the venv to remain isolated for project dependencies (e.g. opencv, numpy, ai libraries), while still being able to access system installed packages like picamera2. 

## (2) Image Processing Layer: `cv2`

Description: OpenCV (cv2) is the primary computer vision library used for image processing, video handling, and preparing data for AI models. It operates on images represented as NumPy arrays and is commonly used for resizing, cropping, color space conversion, drawing overlays (such as bounding boxes), filtering (blur/sharpen/edge detection), and real-time video processing. In this system, OpenCV sits after the camera capture layer (picamera2) and is used to process frames before they are displayed, stored, or passed into an AI model.

Example:
```
import cv2
from picamera2 import Picamera2
...
# Retreive the latest frame from the camera buffer and convert it into a numpy array
frame = camera.capture_array()
# Display the frame until a key is pressed
cv2.imshow("Camera Feed", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Installation Notes:
* To check if opencv is installed: `python -c "import cv2; print('ok')"`
* To find where python is importing it from: `python -c "import cv2; print(cv2.__file__)"`
* To install opencv system wide (recommended) run: `sudo apt update; sudo apt install python3-opencv`
* If you are using a venv, you can install it inside the env with: `pip install opencv-python`

Additional Notes:
* OpenCV depends heavily on NumPy for image data. Images are represented as NumPy arrays, and OpenCV functions operate directly on them. A common issues occures when OpenCV is installed via apt, and NumPy is installed via pip, leading to binary incompatiblity errors because OpenCv is compiled against a specific NumPy version. To fix this, run `sudo apt intall python3-numpy python3-opencv` to ensure they are fully comptabile and tested together. If you are using a venv, likewise, use `pip install numpy opencv-python`. As a general rule of thumb, dont mix apt and pip for core scientific stack unless you manage versions carefully (e.g. numpy, opencv, scipy, matplotlib)

# Virtual Environment Notes

Description: A virtual environment (venv) is an isolated Python environment that allows you to install and manage project-specific libraries without affecting the system-wide Python installation.

1. Create a virtual env with system access: `python -m venv myenv --system-site-packages`
2. Activate the env: `source myenv/bin/activate`
3. Deactivate the env: `deactivate`

What should NOT be installed in the env? Do NOT install system-level hardware or OS-integrated libraries inside the virtual environment, including picamera2, libcamera, system level opencv (apt version), and other os-managed hardware interfaces. 

Note: For most Raspberry Pi camera and AI pipelines, it is best practice to install OpenCV and NumPy at the system level using APT (`sudo apt install python3-opencv python3-numpy`). 
