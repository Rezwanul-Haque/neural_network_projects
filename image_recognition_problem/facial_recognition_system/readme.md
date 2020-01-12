### A face recognition problem can be broken down into the following smaller subproblems
1. Face detection
2. Face recognition

# Face detection
Detect and isolate faces in the image. In an image with multiple faces, we need to detect each 
of them separately. In this step, we should also crop the detected faces from the original input 
image, to identify them separately.

# Face recognition
For each detected face in the image, we run it through a neural network to classify the subject. 
Note that we need to repeat this step for each detected face.