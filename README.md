# FaceRecognitionForEmployees
Detect and Recognize the faces for employees


Face recognition is a technique that involves determining if the face image of any
given person matches with any of the face pictures from tagged stored information.
Face Recognition is natural, straightforward to use and monitor, additionally doesn't
require any special input from the person ahead of the camera. The best means of
doing Face recognition is by opting one-shot learning technique. One-shot learning
aims to learn info regarding object classes from one, or solely a couple of pictures.
The model still must be trained on uncountable information; however, the dataset is
any, however of a similar domain. In a unit of ammunition means of learning, you'll
be able to train a model with any face datasets and use it for your information that is
incredibly less in number. Here we've got used Facenet as a one-shot model.

# Triplet Loss

We have to build a Face Recognition system which will capture real-time images of
employees when they come in front of the camera with the help of computer vision
techniques like OpenCV and model detects the position of the face in the frame
using Haar Cascade then it will recognize that face using Face Recognition model
i.e., Facenet. After we Recognize the face, we get the Name and the present time,
which will be forwarded to EMS and attendance will be marked.

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/CtEzCy03oTmn9n?quality=585&allowAnimation=true)

# Work Flow
![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/oP7MQuEzZMkgZn?quality=480&allowAnimation=true)

# Dataset creation and pre-processing

We collect images of Employees by capturing their images using a regular webcam and store
them with their respective name folder as a class containing face-cropped images. Facial data of
employees are store in such a way that a parent directory consisting of sub-directories. Each
sub-directory is having the name of employees along with employee unique id
(1221_Akshay_Dhomse) and will have facial images (15-20 units) of that respective employee or
user. Following is the dataset Structure:

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/gYzNsWyNMpdORj?quality=636&allowAnimation=true)

# Face Detection

It mainly deals with whether the face is present in the image or not. We need to verify the
person by comparing the detected face with stored face images dataset. We used Haar Cascade
to detect the face in image frame taken from the camera. A detected face is annotated with
bounding box.

# Face Identification


Face Identification

For recognizing or identification purpose, we implement it by using threshold score (an
empirical value), if the score is below the threshold then it is considered positive otherwise
negative. A score is calculated as Euclidean distance between vector embeddings of two faces.
Our network consists of a batch input layer and a deep CNN followed by L2 normalization,
which results in the face embedding. This is followed by the triplet loss during training. The
Triplet Loss minimizes the distance between an anchor and a positive, both of which have the
same identity, and maximizes the distance between the anchor and a negative of a different
identity. It Identifies the person from the detected face in the image by comparing against the
facial database of employees. Once a person gets identified its name is displayed along with the
bounding box.

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/TL0XoaXI5wJxUz?quality=952&allowAnimation=true)

Distance matrix table explains how the Euclidean distance between embeddings are low for two
faces of a similar person and higher for faces of dissimilar person. For our model, we have set
the threshold of 0.7 after verifying the accuracy over multiple people images. Too high or low
threshold score could lead to incorrect recognition. Value is highly dependent on the quality of
facial data.

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/YiOfwHP2U2A9Pm?quality=421&allowAnimation=true)


# Deployment

After successful building the model, we can now move forward for setting up real-time
attendance system. Initially, for recording the attendance of employee, we use to write the CSV
with the following details: id, name, date, time. Later, we added API that will hit the EMS server
for marking attendance

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/O79YbUWQDuE1Gx?quality=601&allowAnimation=true)

![alt text](https://eus-www.sway-cdn.com/s/eDQI1VFHNFZ34TEU/images/AMhbJXF7hj8ro5?quality=480&allowAnimation=true)


# Acknowledgements

![Ajinkya Pathak](https://github.com/Ajinkz)
