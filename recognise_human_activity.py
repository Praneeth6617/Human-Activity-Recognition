import cv2
import numpy as np
from collections import deque

# Parameters
class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt").read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
        self.VIDEO_PATH = ""
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

param = Parameters()
captures = deque(maxlen=param.SAMPLE_DURATION)

print("[INFO] Loading model...")
net = cv2.dnn.readNet(param.ACTION_RESNET)

print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        print("[INFO] Stream ended.")
        break

    frame = cv2.resize(frame, (550, 400))
    captures.append(frame)

    if len(captures) < param.SAMPLE_DURATION:
        continue

    blob = cv2.dnn.blobFromImages(captures, 1.0, (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                  (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.expand_dims(np.transpose(blob, (1, 0, 2, 3)), axis=0)

    net.setInput(blob)
    label = param.CLASSES[np.argmax(net.forward())]

    cv2.rectangle(frame, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow("Human Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()


# ========================RUN COMMAND==========================
# python recognise_human_activity.py
