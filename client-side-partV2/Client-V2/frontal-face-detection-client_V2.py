#import cv2
import time
import json
import socket
import base64
import numpy as np
from threading import Thread
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
#import detect_face
import time
import pickle
import cv2
import io
import socket
import struct
import time
import pickle
import zlib
IP_SERVER = "192.168.1.237"
PORT_SERVER = 8485
TIMEOUT_SOCKET = 10
SIZE_PACKAGE = 4096
DEVICE_NUMBER = 0


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
COLOR_PIXEL = 3 

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('E:/clients/Face_Detection_&_RecognitionV2/client-side-partV2/Client-V2/haarcascade_frontalface_alt.xml')
#face_Cascade = 'E:/clients/client-side-part/haarcascade_frontalface_alt.xml'
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('192.168.1.222', 8485))
# connection = client_socket.makefile('wb')

# while True:
#     video_capture = cv2.VideoCapture(0)
#     video_capture.set(3, IMAGE_WIDTH)
#     video_capture.set(4, IMAGE_HEIGHT)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.237', 8485))
connection = client_socket.makefile('wb')

video_capture = cv2.VideoCapture(DEVICE_NUMBER)
video_capture.set(3, IMAGE_WIDTH)
video_capture.set(4, IMAGE_HEIGHT)
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray1,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw a rectangle around the faces
    cv2.rectangle(frame, (261,174),(457,380),(255,0,255),2)
    result, frame1 = cv2.imencode('.jpg', frame, encode_param)
    data1 = pickle.dumps(frame1, 0)
    size = len(data1)
    print("%%%%%%%%%",size)
    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data1)
    img_counter += 1

    for (x, y, w, h) in faces:
        print(x,y)
        print(x+w,y+h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if(210<int(x)<350 and 150<int(y)<250 and 300<int(x+w)<460 and 310<int(y+h)<450):
            welcome=" Welcome to Infogen labs"
            cv2.putText(frame,welcome, (0, 50), font, 1, (0,0,255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    # result, frame1 = cv2.imencode('.jpg', frame, encode_param)
    # data1 = pickle.dumps(frame1, 0)
    # size = len(data1)
    # print("{}: {}".format(img_counter, size))
    # client_socket.sendall(struct.pack(">L", size) + data1)
    # img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
