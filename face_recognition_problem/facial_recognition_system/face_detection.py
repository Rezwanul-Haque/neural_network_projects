import os

import cv2

face_cascade = cv2.CascadeClassifier('data/pre-trained_cascade/haarcascade_frontalface_default.xml')


# face detection function
def detect_faces(img, draw_box=True):
    # Convert image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(grayscale_img,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE
                                          )

    face_box, face_coords = None, []

    # Draw bounding around detected faces
    for (x, y, width, height) in faces:
        if draw_box:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5)
        face_box = img[y: y + height, x: x + width]
        face_coords = [x, y, width, height]

    return img, face_box, face_coords


if __name__ == "__main__":
    sample_face_dir = 'data/sample_faces'
    detected_face_dir = sample_face_dir + '/detected_faces/'

    files = os.listdir(sample_face_dir)
    images = [file for file in files if 'jpg' in file]

    for image in images:
        img = cv2.imread(sample_face_dir + '/' + image)
        detected_faces, _, _ = detect_faces(img)

        cv2.imwrite(detected_face_dir + image, detected_faces)
    #     cv2.imshow(img, detected_faces)
