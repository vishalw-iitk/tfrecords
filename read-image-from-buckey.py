import gcsfs
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing.image import load_img
image = 'qommunicator/CMLE_Pipeline/Test_words_full/01May_2010_Saturday_tagesschau_default_wort-0/01May_2010_Saturday_tagesschau.avi_fn024802-0.png'
image = image
print("orig_path",image)
fs = gcsfs.GCSFileSystem(project='speedy-aurora-193605',access='full_control')
with fs.open(image) as img_file:
    #img = cv2.imread(img_file, 1)
    img = np.array(load_img(img_file, grayscale = True,target_size=(210,260)))
    print(img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()