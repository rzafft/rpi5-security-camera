import os 

# Supress INFO logs from libcamera (e.g. 0 = DEBUG; 1 = INFO; 2 = WARNING; 3 = ERROR; 4 = FATAL)
os.environ["LIBCAMERA_LOG_LEVELS"] = "3" 


import cv2
import hailo_platform as hailo
import numpy as np

frame = cv2.imread("test_RGB.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

print("\n===== ORIGINAL FRAME =====")
print("type:", type(frame))
print("dtype:", frame.dtype)
print("shape:", frame.shape)
print("min/max:", frame.min(), frame.max())
print("contiguous:", frame.flags['C_CONTIGUOUS'])

processed_frame = cv2.resize(frame, (640, 640))
processed_frame = processed_frame.astype(np.uint8)

print("\n===== PROCESSED FRAME =====")
print("type:", type(processed_frame))
print("dtype:", processed_frame.dtype)
print("shape:", processed_frame.shape)
print("min/max:", processed_frame.min(), processed_frame.max())
print("contiguous:", processed_frame.flags['C_CONTIGUOUS'])

batch_of_processed_frames = np.expand_dims(processed_frame, axis=0)

print("\n===== BATCH OF PROCESSED FRAMES =====")
print("shape:", batch_of_processed_frames.shape)

print("\n===== MEMORY / STRUCTURE =====")
print("original size (bytes):", frame.nbytes)
print("processed size (bytes):", processed_frame.nbytes)
print("batch size (bytes):", batch_of_processed_frames.nbytes)

hef = hailo.HEF("/home/rza/Models/yolov8s.hef")
with hailo.VDevice() as target:
	configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
	network_group = target.configure(hef, configure_params)[0]
	network_group_params = network_group.create_params()
	input_vstream_info = hef.get_input_vstream_infos()[0]
	output_vstream_infos = hef.get_output_vstream_infos()
	input_vstreams_params = hailo.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.UINT8)
	output_vstreams_params = hailo.OutputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hailo.FormatType.FLOAT32)
	with network_group.activate(network_group_params):
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

			input_data = {input_vstream_info.name: batch_of_processed_frames}
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
