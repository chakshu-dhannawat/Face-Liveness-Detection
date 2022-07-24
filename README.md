# Face-Liveness-Detection
 Repository for detection liveness of faces, thus can detect spoofing attacks.<br>

<br>
Steps to run the code:<br>
 To run the plain liveness score detector, clone the repository and run main_code_only_liveness_detection.py<br>
 
 We are not using face_recognition or dlib library of python for liveness score, but the original code in main_code.py uses these libraries to do facial recognition, and identify the person.<br>
 To download dlib library in python follow this [link](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f).<br>
 After dlib is installed, install face_recognition directly via pip.<br>
 
 
 This repository contains a model trained on a CNN backbone and the detects whether the picture provided as input is a spoofed images or a real person. It also supportsvideo input. <br>
 
 
