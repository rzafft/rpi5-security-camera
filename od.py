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
	0: '__background__',
	1: 'person',
	2: 'bicycle',
	3: 'car',
	4: 'motorcycle',
	5: 'airplane',
	6: 'bus',
	7: 'train',
	8: 'truck',
	9: 'boat',
	10: 'traffic light',
	11: 'fire hydrant',
	12: 'stop sign',
	13: 'parking meter',
	14: 'bench',
	15: 'bird',
	16: 'cat',
	17: 'dog',
	18: 'horse',
	19: 'sheep',
	20: 'cow',
	21: 'elephant',
	22: 'bear',
	23: 'zebra',
	24: 'giraffe',
	25: 'backpack',
	26: 'umbrella',
	27: 'handbag',
	28: 'tie',
	29: 'suitcase',
	30: 'frisbee',
	31: 'skis',
	32: 'snowboard',
	33: 'sports ball',
	34: 'kite',
	35: 'baseball bat',
	36: 'baseball glove',
	37: 'skateboard',
	38: 'surfboard',
	39: 'tennis racket',
	40: 'bottle',
	41: 'wine glass',
	42: 'cup',
	43: 'fork',
	44: 'knife',
	45: 'spoon',
	46: 'bowl',
	47: 'banana',
	48: 'apple',
	49: 'sandwich',
	50: 'orange',
	51: 'broccoli',
	52: 'carrot',
	53: 'hot dog',
	54: 'pizza',
	55: 'donut',
	56: 'cake',
	57: 'chair',
	58: 'couch',
	59: 'potted plant',
	60: 'bed',
	61: 'dining table',
	62: 'toilet',
	63: 'tv',
	64: 'laptop',
	65: 'mouse',
	66: 'remote',
	67: 'keyboard',
	68: 'cell phone',
	69: 'microwave',
	70: 'oven',
	71: 'toaster',
	72: 'sink',
	73: 'refrigerator',
	74: 'book',
	75: 'clock',
	76: 'vase',
	77: 'scissors',
	78: 'teddy bear',
	79: 'hair drier',
	80: 'toothbrush'
}


import os 

# Supress DEBUG, INFO, and WARNING logs from libcamera 
# (i.e. 0 = DEBUG; 1 = INFO; 2 = WARNING; 3 = ERROR; 4 = FATAL)
os.environ["LIBCAMERA_LOG_LEVELS"] = "3" 

import hailo_platform as hailo
import cv2
import numpy as np

""" 
===========================================================================================
(1) Load an image into memory
===========================================================================================
"""

# (1.1) Load/store image as a NumPy array
frame = cv2.imread("test_RGB.png")
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
			
			image0_results = results["yolov8s/yolov8_nms_postprocess"][0]
			detections = []
			for group in image0_results:
				for detection in group:
					if len(detection) == 5:
						# (x1,y1) = topleft, (x2,y2) = bottomright
						# if (x1,y1) = (0.21, 0.20), then the top left corner of the detection box is 21% from the left edge of img, and 20% from the top of the img
						x1, y1, x2, y2, confidence = detection
						detections.append((x1, y1, x2, y2, confidence))
				
			

cv2.imshow("Test Image", frame)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
