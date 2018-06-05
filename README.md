# digits-recognition
Recognition multiple ditgits from image using keras, cv2 and mnist (as train data).
+ vgg_keras.ipynb: Use Vgg with Keras to train mnist data for recogize single digit.
+ digits.h5: Model keras is exported from vgg_keras.ipynb.
+ test.py: Use for test: detect rect from image and use model (digits.h5) to predict digits.
## Single line
<img src="./images/photo_1.jpg?raw=true" title="Single Line"/>
<img src="./images/result-photo_1.jpg?raw=true" title="Single Line Result"/>

## Multiple line
<img src="./images/photo_2.jpg?raw=true" title="Multiple Line"/>
<img src="./images/result-photo_2.jpg?raw=true" title="Multiple Line Result"/>
