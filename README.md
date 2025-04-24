# Age-Detection
Age Detection using OpenCV
Real-time Face Detection with Age and Gender Estimation using OpenCV
Libraries and Models
The script uses OpenCV for computer vision tasks, specifically its DNN (Deep Neural Network) module. The key components are: OpenCV, pre-trained models for face detection, age classification, and gender classification.
Pre-trained Models
- **Face detector**: Uses TensorFlow-based `.pb` files (`opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`). This model detects faces in a frame.
- **Age detector**: Uses Caffe-based `.caffemodel` and `.prototxt` files for estimating the age group of detected faces.
- **Gender detector**: Also based on Caffe models, used to predict the gender (Male/Female) of faces.
Code Structure
1. faceBox() Function
The `faceBox()` function detects faces using the loaded `faceNet` model, draws bounding boxes, and returns the frame with annotations as well as the bounding box coordinates of the detected faces.
Steps:
1. A DNN blob is created from the image frame.
2. The blob is passed through the face detection network to get the predictions.
3. Bounding boxes are extracted from the detections, and faces are marked with green rectangles.
2. Model Loading
The following networks are loaded using `cv2.dnn.readNet()`: faceNet, ageNet, and genderNet. These models are used to perform face detection, age classification, and gender classification, respectively.
3. Labels and Mean Values
The script defines label lists for age and gender categories and mean values for pre-processing the image during inference.
4. Video Capture Loop
The main loop captures frames from the webcam and processes them to detect faces and predict age and gender.
Key Steps:
1. Frames are read from the webcam using `cv2.VideoCapture`.
2. The `faceBox()` function detects faces and annotates the frame.
3. Each face is pre-processed (color conversion, histogram equalization, etc.) for age and gender prediction.
4. Age and gender are predicted using the `ageNet` and `genderNet` models.
5. The predicted labels are overlayed on the frame, and the frame is displayed in real-time.
Key Points & Tips
1. **Confidence Threshold**: The confidence threshold for face detection is set to 0.7. This value can be adjusted based on the desired accuracy and recall.
2. **Padding**: Padding is applied around the detected faces to include more context for age and gender classification.
3. **Histogram Equalization**: Used to enhance the image contrast, which may improve prediction accuracy.
4. **Real-time Performance**: Running inference on each frame may be slow. Techniques like resizing the frame or using a GPU can improve performance.
5. **Model Backend**: Consider using CUDA for faster inference on supported GPUs.


Group Members:
ABHI PATEL       KU2407U247
HEVIN            KU2407U293
ARYAN PATEL      KU2407U257
VIRAJ PATEL      KU2407U387
AXAT PATEL       KU2407U261
KALP PATEL       KU2407U308
