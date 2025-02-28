import cv2
import imutils
import Jetson.GPIO as GPIO
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Set up GPIO Pin
GPIO.setmode(GPIO.BOARD)  
GPIO.setup(11, GPIO.OUT)  # Set GPIO pin 11 as output

# Load TensorRT engine File
class TrtModel:
    def __init__(self, engine_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load TensorRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, img):
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        np.copyto(self.inputs[0]["host"], img.ravel())

        # Copy input to GPU
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        self.context.execute_v2(self.bindings)

        # Copy output from GPU
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])
        return self.outputs[0]["host"]

# Load the model.engine
model = TrtModel("Model/build/model.engine", conf_threshold=0.5)

# Define GStreamer pipeline Frame rate and Size
def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=120, flip_method=0):
    return f"v4l2src ! video/x-raw, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! " \
           f"videoconvert ! video/x-raw, format=BGR ! appsink"

# Capture video
pipeline = gstreamer_pipeline()
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = imutils.resize(frame, width=640)

    # Perform inference
    detections = model.infer(frame)

    # Check detections for "Rejected" label and trigger GPIO
    for det in detections:
        if det[4] > model.conf_threshold:  # Assuming confidence score is at index 4
            label = int(det[5])  # Assuming class index is at index 5
            if label == "Rejected":  # Replace with actual class ID for "Rejected"
                print("Rejected label detected!")
                GPIO.output(11, GPIO.HIGH)  # Turn on GPIO pin 11 (active high)
            else:
                GPIO.output(11, GPIO.LOW)   # Turn off GPIO pin 11

    # Display results
    cv2.imshow("Output", frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Cleanup GPIO settings before exiting
