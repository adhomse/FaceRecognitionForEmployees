import cv2
import socket
import base64
import numpy as np

IP_SERVER = "127.0.0.1"
PORT_SERVER = 56810
TIMEOUT_SOCKET = 100
SIZE_PACKAGE = 100000


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
COLOR_PIXEL = 3  # RGB


if __name__ == '__main__':
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connection.settimeout(TIMEOUT_SOCKET)
    connection.connect((IP_SERVER, PORT_SERVER))

    while True:
        try:
            fileDescriptor = connection.makefile(mode='rb')
            result = fileDescriptor.readline()
            fileDescriptor.close()
            result = base64.b64decode(result)
            frame = np.fromstring(result, dtype=np.uint8)
            frame_matrix = np.array(frame)
            frame_matrix = np.reshape(frame_matrix, (IMAGE_HEIGHT, IMAGE_WIDTH,COLOR_PIXEL))
            print(frame_matrix)
            cv2.imshow('Window title', frame_matrix)
            #print(xyz)
            #frame1=cv2.read(frame_matrix)
            #cv2.imshow(frame1)
            #print("yes")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("[Error] " + str(e))

    connection.close()
