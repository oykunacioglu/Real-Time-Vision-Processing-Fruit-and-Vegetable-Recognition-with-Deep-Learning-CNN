import tensorflow as tf
import os
from keras.utils import img_to_array
import numpy as np
import cv2


model = tf.keras.models.load_model("fruit_classifier_model.h5")
print(model.summary())

source_folder ="fruits-360_100x100/fruits-360/Test"
categories = os.listdir(source_folder)
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)

def prepareImage(pathforImage):
    original_image = cv2.imread(pathforImage)
    image_processed = cv2.resize(original_image, (100, 100))
    image_processed = img_to_array(image_processed)     
    image_processed = image_processed / 255.0
    image_processed = np.expand_dims(image_processed, axis=0)
    if original_image is None:
        print(f"\n🚨 Warning: '{pathforImage}' image couldn't be found. Please check the file path/name.")
        return None, None
    return image_processed,original_image
testImagePath = "fruits-360_100x100/fruits-360/Test/Apple Braeburn/0_100.jpg"  
imageForModel,original_image = prepareImage(testImagePath)
if imageForModel is not None:
   resultArray = model.predict(imageForModel,verbose=1)
   answers = np.argmax(resultArray,axis=1)
   predicted_class_index = [answers[0]]
   predicted_fruit = categories[predicted_class_index[0]]
   print("--- Predict Result ---")
   print(f"Predicted Fruit: **{predicted_fruit}**")
   print("---------------------")
   print(answers[0])

   text_to_display = f"Predicted: {predicted_fruit}"
   font = cv2.FONT_HERSHEY_SIMPLEX
   org = (10, 30) 
   fontScale = 0.7
   color = (0, 0, 0) 
   thickness = 2
   
   annotated_image = original_image.copy() 
   cv2.putText(annotated_image, text_to_display, org, font, fontScale, color, thickness, cv2.LINE_AA)
   cv2.imshow("Fruit Prediction", annotated_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
else:
   print("Prediction could not be made because the image could not be loaded.")
   from keras.models import load_model
import numpy as np

model = load_model("model.h5")

# örnek tahmin
prediction = model.predict(np.zeros((1, 100, 100, 3)))
print(prediction)