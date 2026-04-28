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

detections = []

""" 
===========================================================================================
(1) Load an image into memory
===========================================================================================
"""

# (1.1) Load/store image as a NumPy array
frame = cv2.imread("images/test_RGB.png")
# (1.2) Convert the image's color format from BGR to RGB
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

""" 
===========================================================================================
(2) Process the image to prepare it for the model
===========================================================================================
"""

# (2.1) Resize image to model input resolution (YOLOv8s expects 640x640)
processed_frame = cv2.resize(frame, (640, 640))
# (2.2) Convert every pixel value to uint8 (0-255)
processed_frame = processed_frame.astype(np.uint8)
# (2.3) Add batch dimension: (H, W, C) → (1, H, W, C). Models always expect a batch, even if batch size = 1
batch_of_processed_frames = np.expand_dims(processed_frame, axis=0)

print("\n===== ORIGINAL FRAME =====")
print("dtype:", frame.dtype)
print("shape:", frame.shape)
print("min/max:", frame.min(), frame.max())
print("\n===== PROCESSED FRAME =====")
print("dtype:", processed_frame.dtype)
print("shape:", processed_frame.shape)
print("min/max:", processed_frame.min(), processed_frame.max())
print("\n===== BATCH DIMENSION ADDED =====")
print("shape:", batch_of_processed_frames.shape)
print("\n===== MEMORY / STRUCTURE =====")
print("original size (bytes):", frame.nbytes)
print("processed size (bytes):", processed_frame.nbytes)
print("batch size (bytes):", batch_of_processed_frames.nbytes)

""" 
===========================================================================================
# (3) LOAD MODEL + CONFIGURE HAILO DEVICE
===========================================================================================
"""

# (3.1) Load the compiled Hailo Executable Format (HEF) file
hef = hailo.HEF("/home/rza/Models/yolov8s.hef")

# (3.2) Create virtual device (software abstraction of Hailo hardware accelerator)
with hailo.VDevice() as target:
	
	# (3.3) Extract model configuration from the HEF file to create configuration settings for running the neural network on the hailo device
	# 'interface=hailo.HailoStreamInterface.PCIe' specifies that the rapsberry pi and the hailo accelerator will communicate via PCI (high speed hardware communication bus)
	configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
	
	# (3.4) Load the model onto the Hailo device
	# 'target.configure(...)' returns a list of NetworkGroup objects (one per model in HEF file)
	# We select [0] because yolov8s HEF contains one network.
	network_group = target.configure(hef, configure_params)[0]
	
	# (3.5) Generate a runtime paramter container for inference (e.g. stream settings, buffer configuraitons, execution state)
	network_group_params = network_group.create_params()
	
	# (3.6) Extract info about the model's input stream (specifically the first input tensor stream)
	# 'get_input_vstream_infos' returns a metadata about model input streams. It asks the loaded HEF model 'what input streams (tensors) does this model expect?'
	input_vstream_info = hef.get_input_vstream_infos()[0]

	# (3.7)  Extract info about the model's input stream
	# 'get_output_vstream_infos' returns metadata describing model outputs (e.g. detection tensors, bounding boxes, class scores)
	output_vstream_infos = hef.get_output_vstream_infos()

	# (3.8) Creates configuration for sending input data into model.
	# 'network_group' is the loaded model instance
	# 'quantized=True' means the input is treated as quantized integers (0-255) - a common edge for AI acceleration
	# 'format_type=UINT8' means input pixel format is unsigned 8-bit integers (0-255)
	input_vstreams_params = hailo.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.UINT8)

	# (3.9) Create configurations for how model outputs are returned.
	# 'quantized=True' means output originates from quantized hardware representation
	# 'format_type=FLOAT32' means it converts final output into readable floating point values
	output_vstreams_params = hailo.OutputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.FLOAT32)

	# (3.10) Activate the model on the hardware
	# Loads weightsi nto accelerator memory, allocates compute resources, prepares execution pipelines
	with network_group.activate(network_group_params):
		
		# (3.11) Create the full inference pipeline (Input > Model > Output)
		# 'infer_pipeline' will now handle sending tensors to the device, executing inference, and returning results
		with hailo.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
			
			print("\n================ HAILO MODEL INFO ================")
			print("HEF path:", "/home/rza/Models/yolov8s.hef")
			print("\n--- INPUT STREAMS ---")
			for i, info in enumerate(hef.get_input_vstream_infos()):
				print(f"[{i}] name:", info.name)
				print("    shape:", info.shape)
			print("\n--- OUTPUT STREAMS ---")
			for i, info in enumerate(hef.get_output_vstream_infos()):
				print(f"[{i}] name:", info.name)
				print("    shape:", info.shape)
				
			""" 
			===========================================================================================
			(4) Prepare input buffer and run inference
			===========================================================================================
			"""
			
			print("\n================ INFERENCE =================")
			
			# (4.1) Package input tensor into dictionary where key = input stream name, and value = input batch tensor
			input_data = {input_vstream_info.name: batch_of_processed_frames}
			
			 # (4.2) Run inference on hardware accelerator
			results = infer_pipeline.infer(input_data)

			""" 
			===========================================================================================
			(5) Parse output
			===========================================================================================

			When we call 'results = infer_pipeline.infer(input_data)', the yolov8 model scans the image, 
			predicts many possible bounding boxes, assigns a confidence score to each prediction, and 
			applies a filtering step calls 'Non Maximum Supression (NMS). 

			The model often predicts multiple overlapping boxes for hte same boxes. E.g. a single person
			in an image might produce 3 boxes, each with a different confidence score. These boxes overlap 
			heavily, since they all represent the same object. NMS only keeps the best box (highest confidence)
			and removes the overlapping weaker boxes. The result is one clean detection per object instead
			of duplicates.

			Thus, the output of the model looks like this:
			
			results = {
			    "yolov8s/yolov8_nms_postprocess": [
			        image_0_results,
			        image_1_results,
			        ...
			    ]
			}

			Since we only passed one image, results["yolov8s/yolov8_nms_postprocess"][0] = image_0_results = [
			
			    class_0_detections,   # "__background__" (usually empty)
			    class_1_detections,   # "person"
			    class_2_detections,   # "bicycle"
			    class_3_detections,   # "car"
			    ...
			    class_80_detections   # "toothbrush"
			]

			It is important to note that the index in this list is the class id (this matches to the COCO CLASSES dictionary.

			Each class entry is a list of detections. For example, class_1_detections might looks like:

			[
				[x1, y1, x2, y2, confidence],
				[x1, y1, x2, y2, confidence],
				...
			]

			Where (x1,y1) is the top left corner of the bounding box, and (x2,y2) is the bottom left corner. 

			For example, if (x1,y1)=(0.21,0.20), then the top left corner of the detection box is 21% from the 
			left edge of img, and 20% from the top of the img.

			If no objects of that class are found the list should be empty. 
			"""
			
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

h, w = frame.shape[:2]
for y1, x1, y2, x2, conf, label in detections:
	print(f"Detected {label} with {conf:.2f} confidence at ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
	x1 = int(x1 * w)
	x2 = int(x2 * w)
	y1 = int(y1 * h)
	y2 = int(y2 * h)
	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	
				
cv2.imshow("Test Image", frame)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
