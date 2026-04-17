from picamera2 import Picamera2

camera = Picamera2()
config = camera.create_video_configuration(
    main={
      "size": (1920, 1080), 
      "format": "RGB888"
    },
    controls={
      "FrameRate": 30
    }
)
camera.configure(config)

camera.start()

while True:
    frame = camera.capture_array()
    cv2.imshow("Camera", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
