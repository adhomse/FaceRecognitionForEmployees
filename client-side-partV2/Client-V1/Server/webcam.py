#!/usr/bin/env python2
import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face
import time


def main():
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)
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
        video_capture.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    main()