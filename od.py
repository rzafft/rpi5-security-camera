"""
Terminology and Definitions:

1. Inference
2. Tensor
3. Input Stream
4. Ouput Stream
5. Network Group
6. Virtual Device
"""


import os 

# Supress DEBUG, INFO, and WARNING logs from libcamera (i.e. 0 = DEBUG; 1 = INFO; 2 = WARNING; 3 = ERROR; 4 = FATAL)
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

# (2.1) Resize the image to 640x640 px
processed_frame = cv2.resize(frame, (640, 640))
# (2.2) Convert every pixel value to uint8 (0-255)
processed_frame = processed_frame.astype(np.uint8)
# (2.3) Add a new 'batch_size' dimension to the img (batch_size, height, width, channels) so that the model recongizes the input as a batch of 1 image
batch_of_processed_frames = np.expand_dims(processed_frame, axis=0)

print("\n===== ORIGINAL FRAME =====")
print("type:", type(frame))
print("dtype:", frame.dtype)
print("shape:", frame.shape)
print("min/max:", frame.min(), frame.max())
print("contiguous:", frame.flags['C_CONTIGUOUS'])

print("\n===== PROCESSED FRAME =====")
print("type:", type(processed_frame))
print("dtype:", processed_frame.dtype)
print("shape:", processed_frame.shape)
print("min/max:", processed_frame.min(), processed_frame.max())
print("contiguous:", processed_frame.flags['C_CONTIGUOUS'])

print("\n===== BATCH OF PROCESSED FRAMES =====")
print("shape:", batch_of_processed_frames.shape)

print("\n===== MEMORY / STRUCTURE =====")
print("original size (bytes):", frame.nbytes)
print("processed size (bytes):", processed_frame.nbytes)
print("batch size (bytes):", batch_of_processed_frames.nbytes)

""" 
===========================================================================================
(3) Load the model, configure the device, define input/output stream info and configuraiton
===========================================================================================
"""

# (3.1) Load the compiled YoloV8 model file into memory as an object representing a Hailo Executable Format File
# Note: The 'hef' obj contains tne neural network architecture, optimizations for the hailo hardware, and input/output details
hef = hailo.HEF("/home/rza/Models/yolov8s.hef")
# (3.2) Create and open a virtual device (representing the Hailo chip) that can run the model
# Note: Virtual Devices abstract hardware (i.e. our Hailo Chip) so we don't have to manage the hardware directly
with hailo.VDevice() as target:
	# (3.3) Create configuration settings for running the neural network on the hailo device
	# Note: Reads the model (hef) and extract input/output tensor shapes, network layrs, and the hardware exeuction plan
	# Note: Creates configuration settings to tell the Hailo device how it should run the model
	# Note: Specify that the hailo device should run the model using PCIe communication between the hailo chip and the raspberry pi
	configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
	# (3.4) Load the model onto the Hailo device using the configuration settings and the hef obj, and extract the first network group
	# Note: This tells the hailo device 'take the compiled model and set it up on the hardware using these settings'
	# Note: At this oint, the model is being loaded onto the Hailo accelerator and streams+resources are being prepared
	# Note: 'target.configure(...)' returns a list of network groups (since a .hef file can contain one or more neural networks)
	# Note: We retreive the first network group (i.e. ...)[0]) since most Yolo models only have one 
	# Note: 'network_group' now holds the configured, runnable model on the device (i..e it is a fully loaded model instance on the hardware)
	# Note: We will use 'network_group' to create inference pipelines, feed frames into the model, and read detection outputs
	network_group = target.configure(hef, configure_params)[0]
	# (3.5) Generate a runtime paramter container for inference
	# Note: 'network_group_params' is an object that holds things like input/output buffer settinigs, tensor metatdata, stream configuration hooks, and the runtime state needed for execution
	# Note: Think of 'network_group_params' as a prepared settings package for running inference on the model
	network_group_params = network_group.create_params()
	# (3.6) Extract info about the model's input stream (specifically the first input tensor stream)
	# Note: This asks the loaded HEF model 'what input streams (tensors) does this model expect?'
	# Note: 'hef.get_input_vstream_infos()' returns a list of input stream metadata objects (Since some models can have multiple inputs)
	# Note: Each input stream metadata object describes things like input shape (e.g. 640x640x3), data type (unit8), format/layout, and the stream name
	# Note: Since the model only has one input (our image), we get the first input stream metadata object
	# Note: 'input_vstream_info' is NOT the image data - it is information about what the model expects as input
	input_vstream_info = hef.get_input_vstream_infos()[0]
	# (3.7) Extract info about the model's output streams
	# Note: This asks the model 'what outputs does this network produce?'
	# Note: Each output vstream includes something like bounding box predictions, class scores, confidence values, etc.
	output_vstream_infos = hef.get_output_vstream_infos()
	# (3.8) Create configuration settings for feeding input data into the model 
	# Note: This tells the Hailo runtime 'based on this model (network group), create the correct input stream configuration so i can send images into it'
	# Note: 'make_from_network_group()' reads the model's input requirements, builds compatible stream settings, and ensures the input matches what the model expects
	# Note: So instead of manually specifying shapes/types, this derives them from the model
	# Note: 'quantized=True' means that the model expects quantized input (integer based values instead of floats) - a common edge for AI acceleration
	# Note: 'format_type=hailo.FormatType.UNIT8' explicitly sets the input format to uint8
	# Note: This returns a configration object (input_vstreams_params) that defines how input buffers should be structured, the data type, quantization rules, and alightment with model expectations
	input_vstreams_params = hailo.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.UINT8)
	# (3.9) Create configration settings for reading output data from the model after inference
	# Note: This tells the Hailo runtime 'based on this model (network gorup), set up how I should receive and interpret output data
	# Note: 'quantized=True' tells the system that the model outputs may come in quantized form internally - even if the final output is float, the pipeline may still be quantized internally
	# Note: 'format_type=hailo.FormatType.FLOAT32' tells the system to give the final output values as 32bit floading point numbers - even if the hardwar uses quanization internally, we want clean readable results
	output_vstreams_params = hailo.OutputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.FLOAT32)
	# (3.10) Activate the neural network on the hailo device so it is ready to run inference
	# Note: This basically says 'turn the model ON on the hardware using these settings)
	# Note: Literally, this loads the model weights onto the accelerator, prepares input/ouput pipelines, allocates hardware resources, and starts execution context on the device
	with network_group.activate(network_group_params):
		# (3.11) Create and open a full inference pipeline for running the model on the hailo device
		# Note: This sets up the 'data highway' that connects input images to the yolo model running on the hardware to the output results (detections) so we can actually run inference
		# Note: 'hailo.InferVStreams(..)' creates an inference streaming pipline that handles sending input frames into the model, running the model on the hailo accelerator and receiving ouputs back
		# Note: 'hailo.InferVStreams(..)' uses the network group (activated model), input_vstreams_params (rules for how to send images in), and ouput_vstreams_params (rules for how to read results out)
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
				
			print("\n================ INFERENCE =================")

			# (4.1) Create a dictionary that packages input image(s) so they can be fed into the model inference pipeline
			input_data = {input_vstream_info.name: batch_of_processed_frames}
			# (4.2) Run inference (i.e. execute the neural network) on the input image batch and collect outputs
			results = infer_pipeline.infer(input_data)
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
