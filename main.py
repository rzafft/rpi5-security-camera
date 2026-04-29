"""
Terminology and Definitions:

1. Inference:
   Running a trained neural network on a new data (e.g. images) to product predictions

2. Tensor:
   A multi-dimensional array (e.g. image = height × width × channels).

3. Input Stream (VStream Input):
   The structured data channel used to send tensors into the model.

4. Output Stream (VStream Output):
   The structured data channel used to receive predictions from the model.

5. Network Group:
   A fully configured neural network loaded onto the Hailo device.

6. Virtual Device (VDevice):
   A software abstraction representing the physical Hailo accelerator.
"""

# COCO dataset class mapping (model output index → human-readable label)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


import os

# Supress DEBUG, INFO, and WARNING logs from libcamera
# (i.e. 0 = DEBUG; 1 = INFO; 2 = WARNING; 3 = ERROR; 4 = FATAL)
os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

import hailo_platform as hailo
import cv2
import numpy as np
from picamera2 import Picamera2


"""
===========================================================================================
(1) Setup and start the camera
===========================================================================================
"""

from picamera2 import Picamera2
import cv2

camera = Picamera2()
config = camera.create_video_configuration(
	main={
		"size": (640, 640),
		"format": "RGB888"
	},
	controls={
		"FrameRate": 30
	}
)
camera.configure(config)
camera.start()

"""
===========================================================================================
# (2) LOAD MODEL + CONFIGURE HAILO DEVICE
===========================================================================================
"""

# (2.1) Load the compiled Hailo Executable Format (HEF) file
hef = hailo.HEF("/home/rza/Models/yolov8s.hef")

# (2.2) Create virtual device (software abstraction of Hailo hardware accelerator)
with hailo.VDevice() as target:

	# (2.3) Extract model configuration from the HEF file to create configuration settings for running the neural network on the hailo device
	# 'interface=hailo.HailoStreamInterface.PCIe' specifies that the rapsberry pi and the hailo accelerator will communicate via PCI (high speed hardware communication bus)
	configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)

	# (2.4) Load the model onto the Hailo device
	# 'target.configure(...)' returns a list of NetworkGroup objects (one per model in HEF file)
	# We select [0] because yolov8s HEF contains one network.
	network_group = target.configure(hef, configure_params)[0]
	# (2.5) Generate a runtime paramter container for inference (e.g. stream settings, buffer configuraitons, execution state)
	network_group_params = network_group.create_params()

	# (2.6) Extract info about the model's input stream (specifically the first input tensor stream)
	# 'get_input_vstream_infos' returns a metadata about model input streams. It asks the loaded HEF model 'what input streams (tensors) does this model expect?'
	input_vstream_info = hef.get_input_vstream_infos()[0]

	# (2.7)  Extract info about the model's input stream
	# 'get_output_vstream_infos' returns metadata describing model outputs (e.g. detection tensors, bounding boxes, class scores)
	output_vstream_infos = hef.get_output_vstream_infos()

	# (2.8) Creates configuration for sending input data into model.
	# 'network_group' is the loaded model instance
	# 'quantized=True' means the input is treated as quantized integers (0-255) - a common edge for AI acceleration
	# 'format_type=UINT8' means input pixel format is unsigned 8-bit integers (0-255)
	input_vstreams_params = hailo.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.UINT8)

	# (2.9) Create configurations for how model outputs are returned.
	# 'quantized=True' means output originates from quantized hardware representation
	# 'format_type=FLOAT32' means it converts final output into readable floating point values
	output_vstreams_params = hailo.OutputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.FLOAT32)

	# (2.10) Activate the model on the hardware
	# Loads weightsi nto accelerator memory, allocates compute resources, prepares execution pipelines
	with network_group.activate(network_group_params):

		# (2.11) Create the full inference pipeline (Input > Model > Output)
		# 'infer_pipeline' will now handle sending tensors to the device, executing inference, and returning results
		with hailo.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:

			"""
			===========================================================================================
			(3) Capture frames from the camera
			===========================================================================================
			"""

			while True:

				frame = camera.capture_array()

				"""
				===========================================================================================
				(4) Preprocess frame for inference
				===========================================================================================
				"""

				processed_frame = cv2.resize(frame, (640, 640))
				processed_frame = processed_frame.astype(np.uint8)
				batch_of_processed_frames = np.expand_dims(processed_frame, axis=0)

				"""
				===========================================================================================
				(5) Prepare input buffer and run inference
				===========================================================================================
				"""

				# (4.1) Package input tensor into dictionary where key = input stream name, and value = input batch tensor
				input_data = {input_vstream_info.name: batch_of_processed_frames}

				# (4.2) Run inference on hardware accelerator
				results = infer_pipeline.infer(input_data)

				"""
				===========================================================================================
				(6) Parse output / capture detections
				===========================================================================================
				"""

				detections = []
				image_results = results["yolov8s/yolov8_nms_postprocess"]
				for i, image_result in enumerate(image_results):
					for j, class_detections in enumerate(image_result):
						coco_class = COCO_CLASSES[j]
						for k, detection in enumerate(class_detections):
							if len(detection) == 5:
								y1, x1, y2, x2, confidence = detection
								if confidence < 0.5:
									continue
								detections.append((y1, x1, y2, x2, confidence, coco_class))

				"""
				===========================================================================================
				(7) Draw detections on frame and show output
				===========================================================================================
				"""

				h, w = frame.shape[:2]
				for y1, x1, y2, x2, conf, label in detections:
					print(f"Detected {label} with {conf:.2f} confidence at ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
					x1 = int(x1 * w)
					x2 = int(x2 * w)
					y1 = int(y1 * h)
					y2 = int(y2 * h)
					cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

				cv2.imshow("Camera", frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

camera.stop()
cv2.destroyAllWindows()
