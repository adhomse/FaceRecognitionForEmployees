from sklearn.metrics.pairwise import pairwise_distances #To find out Pairwise distance between x, y[] matrix
from tensorflow.python.platform import gfile # its file I/O wrapper without thread loacking
from scipy import misc
import tensorflow as tf
import numpy as np
import detect_and_align
import argparse
import time
import cv2
import os
import csv
import re
from datetime import datetime
from itertools import tee, islice, chain
import socket
import base64
#import numpy as np

class IdData():
    """We Created the class to Keeps track of known identities and calculates id matches"""

    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder,
                 phase_train_placeholder, distance_treshold):
        print('Loading known identities: ', end='')
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []

        image_paths = []
        image_path1 = []
        ids1 = os.listdir(os.path.expanduser(id_folder))
        for id_name in ids1:
            
            id_dir = os.path.join(id_folder, id_name)
            image_path1 = image_path1 + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]
        for i in image_path1:
            num=re.sub(r'\\',r"/",i)
            image_paths.append(num)    #image_paths=image_paths.append(num)

            #image_paths=[r'C:/myproject/Copy_FaceRecognition-master/ids/Ajinkya/Ajinkya0.png', r'C:/myproject/Copy_FaceRecognition-master/ids/Akshay/Akshay0.png', r'C:/myproject/Copy_FaceRecognition-master/ids/Kirti/Kirti0.png',r'C:/myproject/Copy_FaceRecognition-master/ids/Diksha/Diksha0.png',r'C:/myproject/Copy_FaceRecognition-master/ids/Vishal/Vishal0.png',r'C:/myproject/Copy_FaceRecognition-master/ids/Neha/Neha0.png',r'C:/myproject/Copy_FaceRecognition-master/ids/Rahul/Rahul0.png',r'C:/myproject/Copy_FaceRecognition-master/ids/Rahul/Rahul1.png']

        print('Found %d images in id folder' % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = misc.imread(os.path.expanduser(image_path), mode='RGB')
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print("Warning: Found multiple faces in id image: %s" % image_path +
                      "\nMake sure to only have one face in the id images. " +
                      "If that's the case then it's a false positive detection and" +
                      " you can solve it by increasing the thresolds of the cascade network")
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            self.id_names += [image_path.split('/')[-2]] * len(face_patches)

        return np.stack(aligned_images), id_image_paths #Join a sequence of arrays along a new axis.

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split('/')[-1] for path in id_image_paths] # To get the name of the image from the path reverse the path and get the final folder name from the path C:/myproject/Copy_FaceRecognition-master/ids/Ajinkya/Ajinkya0.png
        print('Distance matrix:\n{:20}'.format(''), end='')
        [print('{:20}'.format(name), end='') for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print('\n{:20}'.format(path), end='')
            for distance in distance_row:
                print('{:20}'.format('%0.3f' % distance), end='')
        print()

    def find_matching_ids(self, embs):
        matching_ids = []
        matching_distances = []
        distance_matrix = pairwise_distances(embs, self.embeddings)
        print("Embeddings",embs,self.embeddings)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row) #Returns the indices of the minimum values along an axis.
            if distance_row[min_index] < self.distance_treshold: # if min_index is less then the threshold i.e., 1.2 
                matching_ids.append(self.id_names[min_index]) # If true then append the name into maching_ids[] list 
                matching_distances.append(distance_row[min_index])# and also append the distance associated with that image
            else:
                matching_ids.append(None)
                matching_distances.append(None)
        return matching_ids, matching_distances  # find_matching_ids returns the name and the distance of that perticular image


def load_model(model): 
    model_exp = os.path.expanduser(model) #pass the facenet model path
    if (os.path.isfile(model_exp)):
        print('Loading model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:  # 'gfile' make it more efficient for some backing filesystems.
            graph_def = tf.GraphDef() #'GraphDef' is the class created by the protobuf liberary
            graph_def.ParseFromString(f.read()) 
            tf.import_graph_def(graph_def, name='') # To load the TF Graph
    else:
        raise ValueError('Specify model file, not directory!')


def main(args):
    print('################################################################################')
    my_style_data=[]
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Setup models
            mtcnn = detect_and_align.create_mtcnn(sess, None) #It calls create_mtcnn function from the detect_and_align file 

            load_model(args.model) #IT loads the facenet 20170512-110547.pb pre-trained model
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load anchor IDs
            id_data = IdData(args.id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, args.threshold)
            #url ='rtsp://192.168.137.135:4747/video'

            #cap = cv2.VideoCapture(0)
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
                    xyz=cv2.imshow('Window title', frame_matrix)
                    cv2.read(xyz)
                    print("cccc")
            

                    frame_height = frame_matrix.get(cv2.CAP_PROP_FRAME_HEIGHT)


                    show_landmarks = True
                    show_bb = True
                    show_id = True
                    show_fps = False
                    show_bb1 = True
                    while(True):
                        start = time.time()
                        v_offset = 50 
                        time.sleep(0.0001)
                        _, frame = frame_matrix.read()
                        frame1=frame

                        # Locate faces and landmarks in frame
                        face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

                        if len(face_patches) > 0:
                            face_patches = np.stack(face_patches)
                            feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                            embs = sess.run(embeddings, feed_dict=feed_dict)

                            print('Matches in frame:')
                            matching_ids, matching_distances = id_data.find_matching_ids(embs)

                            for bb, landmark, matching_id, dist in zip(padded_bounding_boxes, landmarks, matching_ids, matching_distances):
                                if matching_id is None:
                                    matching_id = 'Unknown'
                                    print('Unknown! Couldn\'t fint match.')
                                else:
                                    #if(int(bb[0])<=170 and int(bb[3])<=357 and int(bb[2])<=437 and int(bb[1])<=164):
                                    print('Hi akki %s! Distance: %1.4f' % (matching_id, dist))
                                    now=datetime.now()
                                    
                                    #csvData = [matching_id, dist,now.strftime("%x %I:%M:%S %p")]
                                              
                                    '''with open('C:/myproject/Phase2-Copy_FaceRecognition-master/Student4.csv', 'a') as csvFile:
                                        writer = csv.writer(csvFile)
                                        writer.writerow(csvData)

                                    csvFile.close()'''

                                    


                                if show_id:                            
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    print("bb_Frame",bb)
                                    if(210<int(bb[0])<350 and 150<int(bb[1])<250 and 300<int(bb[2])<460 and 310<int(bb[3])<450):

                                        welcome=" Welcome to Infogen labs"
                                        cv2.putText(frame,matching_id+ welcome, (0, 50), font, 1, (0,0,255), 2, cv2.LINE_AA)
                                        cv2.putText(frame,matching_id+now.strftime(" %I:%M%p"), (bb[0], bb[3]), font, 1, (0,0,255), 2, cv2.LINE_AA)
                                        #csvData = [matching_id, dist,now.strftime("%x  %I:%M:%S %p")]
                                       

                                        def previous_and_next(some_iterable):
                                            prevs, items, nexts = tee(some_iterable, 3)
                                            prevs = chain([None], prevs)
                                            nexts = chain(islice(nexts, 1, None), [None])
                                            return zip(prevs, items, nexts)
                                        csvData = [matching_id, dist,now.strftime("%x %I:%M:%S %p")]
                                        with open('C:/myproject/1Copy_FaceRecognition-master - Copy/Student5.csv', 'a') as csvFile1:
                                            writer = csv.writer(csvFile1)
                                            writer.writerow(csvData)
                                        csvFile1.close()
                                        my_style_data.append(csvData)
                                        print('******************************************************************************************')
                                        print(my_style_data)
                                        print('******************************************************************************************')
                                        if len(my_style_data) >=7:
                                            for prevs, item, nxt in previous_and_next(my_style_data):
                                                try:
                                                    if prevs[0] != item[0] or item[0] != nxt[0]:
                                                        
                                                        with open('C:/myproject/1Copy_FaceRecognition-master - Copy/Employee6.csv', 'a') as csvFile:
                                                            writer = csv.writer(csvFile)
                                                            
                                                            writer.writerow(item)
                                                        csvFile.close()
                                                except: pass
                                            del(my_style_data[0:7])




                                        '''if(matching_id=="Akshay"):
                                            with open('C:/myproject/Phase2-Copy_FaceRecognition-master/Akshay.csv', 'a') as csvFile:
                                                writer = csv.writer(csvFile)
                                                writer.writerow(csvData)
                                            csvFile.close()
                                        if(matching_id=="Ajinkya"):
                                            with open('C:/myproject/Phase2-Copy_FaceRecognition-master/Ajinkya.csv', 'a') as csvFile:
                                                writer = csv.writer(csvFile)
                                                writer.writerow(csvData)
                                            csvFile.close()'''



                                if show_bb:
                                    cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 1)
                                if show_bb1:
                                    cv2.rectangle(frame1, (261,174),(457,380), (255,0,255),2)
                                if show_landmarks:
                                    for j in range(5):
                                        size = 1
                                        x=(int(landmark[j]))
                                        top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
                                        bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
                                        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                        else:
                            print('Couldn\'t find a face')

                        

                        

                        cv2.imshow('frame', frame)

                        key = cv2.waitKey(100)
                        if key == ord('q'):
                            break
                        elif key == ord('l'):
                            show_landmarks = not show_landmarks
                        elif key == ord('b'):
                            show_bb = not show_bb
                        elif key == ord('i'):
                            show_id = not show_id
                        elif key == ord('f'):
                            show_fps = not show_fps
                    frame_matrix.release()
                    cv2.destroyAllWindows()
if __name__ == '__main__':
        parser = argparse.ArgumentParser() #To access the arguments from the command line

    parser.add_argument('model', type=str, help='C:/myproject/Copy_FaceRecognition-master/20170512-110547/20170512-110547.pb')
    parser.add_argument('id_folder', type=str, nargs='+', help='C:/myproject/Copy_FaceRecognition-master/ids/')
    parser.add_argument('-t', '--threshold', type=float,
        help='Distance threshold defining an id match', default=1.2)
    main(parser.parse_args()) #parse_args() will typically be called with no arguments, and the 'ArgumentParser' will automatically determine the command-line arguments
