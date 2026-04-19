# rpi5-security-camera

Goal: Using a raspberry pi, create a secuirty camera system. It will stream footage live, and use ai to detect objects and people. Whenever it finds someone, or something, it should print it to the terminal!


<br>
<br>

# Setup

### Hardware 
1. Core: Raspberry Pi 5 (8Gb Ram)
2. Storage: Raspberry Pi NVMe M.2 SSD (512Gb)
3. Storage Interface: ACASIS NVMe to USD Adapter (10 Gbps)
4. AI Acceleration: Raspberry Pi AI Hat+ (Hailo 8 Accelerator with 26 Tops)
5. Vision Input: Raspberry Pi Camera Module 3

### Additional Information
* Operating System: Raspberry Pi OS
* Python Installation: `\usr\bin\python3`

<br>
<br>

<br><br>

# Picking a Object Detection Model (Downloading YOLOv8 .hef file) 

For this project we will use the YOLOv8 object detection model. We will download it from the Hailo Model Zoo, a collection of ready-to-use models that are already compiled in .hef format. 

Go to Hailo's official model zoo at https://github.com/hailo-ai/hailo_model_zoo. In README.md, you can find links to object detection models specifically for the Hailo 8 chip (compatibility matters). You can also just go to docs/public_models/hailo8/hail08_object_detection.rst. Here you will find a table where each row is a object detection model for the hailo 8 chip that was compiled using the hailo dataflow compiler (DFC). We should see the following rows:

* yolov8x⭐: slowest, most accurate
* yolov8l⭐: slow, accurate
* yolov8m⭐: balenced speed and accuracy
* yolov8s⭐: faster, less accurate 
* yolov8n⭐: fastst, least accurate

Choose the yolov8s varient. This is the 'small' version of YOLOv8. It should have the following metrics. 

* Network Name: yolov8s⭐
* Float Mean Average Precision / Accuray before optimization (floatmAp): 44.6
* Hardware Mean Average Precision / Accuracy after compiling for Hailo (HardwaremAP): 44.0
* FPS (Batch size = 1) / How many frames can the chip process per second: 491
* FPS (batch size = 8) / How many frames can the chip process at once: 491
* Input resolution (HxWxC) / What width/height/channles does it expect: 640x640x3
* Params (M) / How many parameters does the model have: 11.2
* OPS (G) / How many operations are required: 28.6

When you click the 'HEF' download link in the row, you should download a file named 'yolov8ls.hef'

### Using the model

HailoRT expacts data format to match the model exactly. Therefore, every input we send to the model must be...

1. 640 x 640 x 640 (height, width, channels)
2. RGB
3. uint8 (0 - 255)

Thus, each input should be a (640, 640, 3) NumPy array with dtype=uint8. 

In order to use the model we must...

1. Load .hef
2. Configure device
3. Create input/output streams
4. Run inference

HailoRT then returns raw tensors. You'll need to decode the output into boxes, filter confidence thresholds, and remove duplicates. 

<br>
<br>

# Understanding the AI Hat+

The Raspberry Pi AI HAT+ is a hardware accelerator board. Inside it is a Hailo-8 chip with 26 tops - a neural network inference processor that can run about 26 trillion AI operations per second. Inference processors like this are used to run AI models, not train them - that typically happens on GPUs in data centers. These chips are designed for efficent AI at the edge, meaning that AI runs locally on the device instead of sending data to the cloud. 

So, in simple terms, the AI HAT+ is a dedicated AI co-processor that sits next to the Raspberry Pi and runs neural networks extremely fast while the Pi handles everything else. It is not a GPU, not a training chip, and not something you program like a CPU. It is a model execution engine for vision AI.

The AI Hat+ is fully integrated into the Raspberry Pi camera software stack (waht does this mean)

Hailo's stack is simply:

* DFC (dataflow compiler): Builds .hef files
* HailoRT: Runs .hef files
* Model Zoo: Download precompiled .hef files 
* TAPPAS: This is the full application pipeline. It is a prebuilt pipeline that combines camera input, HailoRT inference, post processing (drawing boxes, lables, etc.), and outputs to the screen.  

### The Hailo Dataflow Compiler (DFC)

The hailo chip is not like a GPU which runs general purpose code (CUDA, etc.). Instead, it runs a precompiled neural network pipeline. So, you must compile it first. That is where the DFC comes in. 

The DFC is the thing that turns an AI model into something the Hailo chip can actually run. It is a model conversion + optimization toolchain. It takes a model like TensorFlow, PyTorch, TFLite, etc., and converts it into a Hailo specific binary called a HEF file (Hailo Executable Format). The .hef file is what the Hailo 8 accelerator actually runs... So in full, it..

Thus, we could train our own model, compile it with DFC to get a .hef file, and deploy it!

IMPORTANT NOTE: You usually DO NOT run the compiler on the raspberry pi. DFC runs on x86 architecture. So you should run it in a x86 linux machine then copy the .hef to your pi

You can download the hailo dataflow compiler to run AI accplications at hailo.ai/developer-zone. 

### HailoRT

Think of HailoRT as the driver + api that feeds data into the hailo chip and pulls results out. Thus, we talk to the hailo chip via HailoRT. It is the hailo runtime. It runs .hef files. 

To install hailort, follow these steps:
1. Go to the Hailo Developer Zone at https://hailo.ai/developer-zone/. Note, i created an account (ryanzafft@gmail.com)
2. Go to the downloads page and filter on the hailo8/8l products, hailort package only, arm64 architecture, linux os, and python version 3.11 (run 'python3 --version' to check). You should see hailort-pcie-driver_4.23.0_all.deb (the kernel level driver for pcie devices), hailort-4.23.0-cp311-cp311-linux_aarch64.whl (the python bindings), and hailort_4.23.0_arm64.deb (the core runtime library).
3. Install the runtime ('sudo apt install ./hailort_4.23.0_arm64.deb'), then run the following commands

```
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Note, selecting 'hailort' instead of './hailort_4.23.0_arm64.deb'
The following packages were automatically installed and are no longer required:
  libbasicusageenvironment1 libgroupsock8 liblivemedia77 linux-headers-6.12.25+rpt-common-rpi
  linux-headers-6.12.25+rpt-rpi-2712 linux-headers-6.12.25+rpt-rpi-v8 linux-image-6.12.25+rpt-rpi-2712
  linux-image-6.12.25+rpt-rpi-v8 linux-kbuild-6.12.25+rpt python3-v4l2
Use 'sudo apt autoremove' to remove them.
The following NEW packages will be installed:
  hailort
0 upgraded, 1 newly installed, 0 to remove and 4 not upgraded.
Need to get 0 B/6,878 kB of archives.
After this operation, 0 B of additional disk space will be used.
Get:1 /home/rza/Downloads/hailort_4.23.0_arm64.deb hailort arm64 4.23.0 [6,878 kB]
Selecting previously unselected package hailort.
(Reading database ... 172813 files and directories currently installed.)
Preparing to unpack .../hailort_4.23.0_arm64.deb ...
Unpacking hailort (4.23.0) ...
Setting up hailort (4.23.0) ...
Do you wish to activate hailort service? (required for most pyHailoRT use cases) [y/N]: 
Stopping hailort.service
N: Download is performed unsandboxed as root as file '/home/rza/Downloads/hailort_4.23.0_arm64.deb' couldn't be accessed by user '_apt'. - pkgAcquire::Run (13: Permission denied)


rza@rp5-pios:~ $ lspci
0001:00:00.0 PCI bridge: Broadcom Inc. and subsidiaries BCM2712 PCIe Bridge (rev 21)
0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)
0002:00:00.0 PCI bridge: Broadcom Inc. and subsidiaries BCM2712 PCIe Bridge (rev 21)
0002:01:00.0 Ethernet controller: Raspberry Pi Ltd RP1 PCIe 2.0 South Bridge
rza@rp5-pios:~ $ systemctl status hailort.service
● hailort.service - HailoRT service
     Loaded: loaded (/lib/systemd/system/hailort.service; enabled; preset: enabled)
     Active: active (running) since Sat 2026-04-18 18:59:41 MDT; 14min ago
       Docs: https://github.com/hailo-ai/hailort
    Process: 690 ExecStart=/usr/local/bin/hailort_service (code=exited, status=0/SUCCESS)
    Process: 749 ExecStartPost=/bin/sleep 0.1 (code=exited, status=0/SUCCESS)
   Main PID: 744 (hailort_service)
      Tasks: 11 (limit: 9572)
        CPU: 70ms
     CGroup: /system.slice/hailort.service
             └─744 /usr/local/bin/hailort_service

Apr 18 18:59:41 rp5-pios systemd[1]: Starting hailort.service - HailoRT service...
Apr 18 18:59:41 rp5-pios systemd[1]: Started hailort.service - HailoRT service.
rza@rp5-pios:~ $ hailort scan
bash: hailort: command not found
rza@rp5-pios:~ $ 


```
```
1. lsmod: list loaded kernel modules (which drivers are active). If you see '0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)', then the hailo chip is physically detected, and we likley dont need to install the driver

2. dmesg | grep hailo: Print all the kernel messages (system logs) that contain 'hailo', to check if a device list the hailo chip was detected

3. hailortcli fw-control identify: Query the hailo device firmware and confirm it is detected

4. which hailortcli: Show where the command lives (you should see /usr/bin/hailortcli)

5: lspci | grep -i hailo: confirm device is visible

6. systemctl status hailort.service: check service (want to see active running)

7. hailortcli scan: Should return device buss address. If you see "hailo devices not found', then the os sees teh pcie device but hailort cannot open it.

```

4. Run 'lsmod' to make sure the hailo chip is detected via pcei, then run 'which hailortcli' to make sure hailortcli is in /usr/bin, and then run 'hailortcli scan'. If you see "hailo devices not found', then the os sees teh pcie device but hailort cannot open it. Run 'ls /dev | grep hailo'. If you see nothing then we still need the PCIe driver. E.g. run 'sudo apt install ./hailort-pcie-driver_4.23.0_all.deb'

5. Install the pcei driver 

<br>
<br>

# Required Libraries

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

<br>
<br>

# Virtual Environment Notes

Description: A virtual environment (venv) is an isolated Python environment that allows you to install and manage project-specific libraries without affecting the system-wide Python installation.

1. Create a virtual env with system access: `python -m venv myenv --system-site-packages`
2. Activate the env: `source myenv/bin/activate`
3. Deactivate the env: `deactivate`

What should NOT be installed in the env? Do NOT install system-level hardware or OS-integrated libraries inside the virtual environment, including picamera2, libcamera, system level opencv (apt version), and other os-managed hardware interfaces. 

Note: For most Raspberry Pi camera and AI pipelines, it is best practice to install OpenCV and NumPy at the system level using APT (`sudo apt install python3-opencv python3-numpy`). 

