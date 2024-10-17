import cv2
import uuid
import os
import time

# menentukan jenis-jenis tingkat matangan
labels = ['Hai', 'Peace']
number_images = 5

# folder PATH
IMAGES_PATH = "Tensorflow/workspace/images/collectedimages"

# Check if IMAGES_PATH exists, if not create it
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

# Loop through each label and create a folder for it if it doesn't exist
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)


for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_images):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()