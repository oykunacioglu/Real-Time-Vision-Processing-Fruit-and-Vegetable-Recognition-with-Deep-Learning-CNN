
This project is an  object detection and classification via webcam using Convolutional Neural Networks (CNN) and Computer Vision. By utilizing the Fruit-360 dataset, the model has been trained and stabilized to perform reliably under real-world conditions.

###🧠 Model Architecture & Training
The core of the system is a deep learning model designed for autonomous feature extraction from raw pixels:

Architecture: A deep neural network consisting of Conv2D, MaxPooling2D, and Dropout layers with a 100x100x3 input size.

Performance: The training process reached a 98.36% Validation Accuracy and a 0.0645 Validation Loss over 42 epochs.

Optimization: An EarlyStopping mechanism was implemented to prevent overfitting, ensuring the most successful weights were preserved for production.

###🛠️ Real-Time Processing & Optimization
To ensure the system remains stable outside of a laboratory environment, the following engineering solutions were implemented:

##🎯 ROI (Region of Interest) Filtering: A 280x280 pixel focus area was created to keep the model's focus on the center of the frame and eliminate background noise.

##🛑 Confidence Thresholding: A 70% confidence threshold was applied to prevent random guesses; the system only presents results to the user when it is statistically certain.

##🗺️ Heuristic Label Mapping: A name_fixer dictionary was developed to synchronize the model's internal indices with the local file system (e.g., mapping "Zucchini 1" to "Green Apple").

###💻 Technologies Used
Python 3.x

TensorFlow & Keras: Deep learning model design and training.

OpenCV: Camera streaming, ROI visualization, and real-time image processing.

NumPy: Matrix operations and data normalization.


<img width="1275" height="958" alt="image" src="https://github.com/user-attachments/assets/7632747f-fd38-4484-91c4-01b6d8cc8539" />

