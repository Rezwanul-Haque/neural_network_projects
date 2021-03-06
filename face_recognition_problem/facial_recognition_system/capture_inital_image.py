import math
import os

import cv2

import face_detection
import utils

video_capture = cv2.VideoCapture(0)
counter = 5

dir_path = os.path.dirname(os.path.realpath(__file__))

while True:
    _, frame = video_capture.read()
    frame, face_box, face_coords = face_detection.detect_faces(frame)
    text = 'Image will be taken in {}..'.format(math.ceil(counter))
    if face_box is not None:
        frame = utils.write_on_frame(frame, text, face_coords[0], face_coords[1] - 10)
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    counter -= 0.1
    if counter <= 0:
        cv2.imwrite(dir_path + f'/data/db_images/true_img.png', face_box)
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Onboarding Image Captured")
