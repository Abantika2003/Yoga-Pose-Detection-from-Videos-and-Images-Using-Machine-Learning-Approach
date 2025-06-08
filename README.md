# Yoga-Pose-Detection-from-Videos-and-Images-Using-Machine-Learning-Approach
AI and ML project
Yoga Pose Detection Using GANs and Real-Time Pose Estimation
📌 Introduction
This project implements an advanced Yoga Pose Detection system using a combination of Generative Adversarial Networks (GANs) for data augmentation, state-of-the-art pose estimation models, and traditional machine learning classifiers for pose classification. The system supports both image-based input and real-time webcam feed for live yoga pose detection and accuracy evaluation.

🎯 Objective
The primary objective of this project is to develop an intelligent system that can:

Recognize and classify various yoga poses from static or real-time images.

Evaluate the correctness and accuracy of the detected pose.

Provide feedback to users practicing yoga at home or in training environments.

🧠 Key Components
1. Input Image / Real-Time Webcam Feed
Users can input images or use webcam integration for real-time yoga pose detection.

2. GAN Augmentation
Utilizes GANs to generate diverse augmented images to enrich the dataset.

Helps to overcome dataset limitations and improve model generalization.

3. Key Points Detection
Extracts body keypoints using advanced models like:

OpenPose

PoseNet

PIFPAF

4. Pose Estimation
Predicts 2D and 3D human poses using:

YOLOv8 for 2D pose estimation

SSD (Single Shot Detector) for 3D pose estimation

5. Pose Classification
Classified using machine learning models:

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naïve Bayes

Logistic Regression

Random Forest

6. Evaluation Metrics
Models are evaluated using:

Accuracy

Precision

Recall

F1-Score

7. Pose Accuracy Feedback
Real-time feedback on how accurately a user is performing a yoga pose.

Displays pose names, detected confidence levels, and visual feedback.

🖥️ Workflow Overview
Input image or webcam frame is processed.

Data augmentation is applied using GANs.

Keypoints are detected and passed to pose estimation models.

Classified using ML models.

Real-time feedback and pose accuracy are displayed.

📷 Sample Output
Real-time detection displays:

Yoga pose name

Pose confidence score

Visual skeleton overlay

🔍 Tools and Technologies Used
Python

OpenPose / PoseNet / PIFPAF

YOLOv8, SSD

GANs for augmentation

scikit-learn (SVM, KNN, Naïve Bayes, etc.)

OpenCV for webcam integration

Matplotlib / Plotly for 3D pose visualization

📊 Future Scope
Incorporation of feedback loop for posture correction.

Integration with mobile apps.

Support for more yoga poses and multi-person tracking.
