import cv2
import sys
import logging as log
import datetime as dt
import time 
import math
import argparse

def face_box(net, frame, conf_threshold= 0.7):
    frame_dnn = frame.copy()
    height = frame_dnn.shape[0]
    width = frame_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    face_detect = net.forward()
    boxess = []
    for i in range(face_detect.shape[2]):
        con = face_detect[0, 0, i, 2]
        if con > conf_threshold:
            a1 = int(face_detect[0, 0, i, 3] * width)
            b1 = int(face_detect[0, 0, i, 4] * height)
            a2 = int(face_detect[0, 0, i, 5] * width)
            b2 = int(face_detect[0, 0, i, 6] * height)
            boxess.append([a1, b1, a2, b2])
            cv2.rectangle(frame_dnn, (a1, b1), (a2, b2), (0, 255, 0), int(round(height/150)), 8)
    return frame_dnn, boxess

    capturee.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Use this script to run gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
args = parser.parse_args()

fproto = "opencv_face_detector.pbtxt"
fmodel = "opencv_face_detector_uint8.pb"

gproto = "gender_deploy.prototxt"
gmodel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genlist = ['Male', 'Female']

# Load network
gen_net = cv2.dnn.readNet(gmodel, gproto)
face_net = cv2.dnn.readNet(fmodel, fproto)

capturee = cv2.VideoCapture(args.input if args.input else 0)
padding = 20
while cv2.waitKey(1) < 0:
    t = time.time()
    has_frame, frame = capturee.read()
    if not has_frame:
        cv2.waitKey()
        break

    frameFace, boxess = face_box(face_net, frame)
    if not boxess:
        print("No face Detected, Checking next frame")
        continue

    for bbox in boxess:
        print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gen_net.setInput(blob)
        genderPreds = gen_net.forward()
        gender = genlist[genderPreds[0].argmax()]

        label = "{}".format(gender)
        #Displaying Gender 
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
        #Name of dialog box 
        cv2.imshow("GenderDetection", frameFace)