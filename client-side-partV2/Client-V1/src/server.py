import cv2
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
import detect_face
import time

SERVER_IP = "127.0.0.1"
SERVER_PORT = 56810
MAX_NUM_CONNECTIONS = 20
DEVICE_NUMBER = 0

class ConnectionPool(Thread):

    def __init__(self, ip_, port_, conn_, device_):
        Thread.__init__(self)
        self.ip = ip_
        self.port = port_
        self.conn = conn_
        self.device = device_
        print("[+] New server socket thread started for " + self.ip + ":" +str(self.port))

    def run(self):
        try:
            while True:
                ret, frame = self.device.read()
                a = b'\r\n'
                data = frame.tostring()
                da = base64.b64encode(data)
                self.conn.sendall(da + a)

        except Exception as e:
            print("Connection lost with " + self.ip + ":" + str(self.port) +"\r\n[Error] " + str(e.message))
        self.conn.close()

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(DEVICE_NUMBER)
    video_capture.set(3, 640)
    video_capture.set(4, 480)
    frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    minsize = 25 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor


    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        show_landmarks = True
        show_bb = True
        show_id = True
        show_fps = False
        show_bb1 = True
        while(True):
            start = time.time()
            v_offset = 50
            time.sleep(0.0001)
            ret, frame = video_capture.read()
            frame1=frame

            if not ret:
                break
            # Display the resulting frame
            
            img = frame[:,:,0:3]
            boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            
            

            print(boxes)
            if show_bb:
                for i in range(boxes.shape[0]):
                    pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                    pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                    cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
                    
                    cv2.imshow('Video', frame)
                

                
                    
            if show_bb1==True:
                print("akshay")
                cv2.rectangle(frame, (261,174),(457,380),(255,0,255),2)

                cv2.imshow('Video', frame1)

            key = cv2.waitKey(200)
            if key == ord('q'):
                break
            #elif key == ord('b'):
                #show_bb == show_bb

            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        
        
    print("Waiting connections...")
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.bind((SERVER_IP, SERVER_PORT))
    connection.listen(MAX_NUM_CONNECTIONS)
    while True:
        (conn, (ip, port)) = connection.accept()
        thread = ConnectionPool(ip, port, conn, cap)
        thread.start()
    connection.close()
    video_capture.release()
    cv2.destroyAllWindows()
