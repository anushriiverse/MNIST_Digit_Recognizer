import cv2
import numpy as np

def center_and_resize(img):
    # find non-zero digits
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((28,28), dtype=np.uint8)

    # prepare a bounding box
    x, y, w, h = cv2.boundingRect(coords)
    # crop the image to the bounding box
    cropped = img[y:y+h, x:x+w]

    # resize the image to 20x20 while maintaining aspect ratio
    if w > h:
        new_w = 20
        new_h = int(20 * (h/w))
    else:
        new_h = 20
        new_w = int(20 * (w/h))

    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # paste in the centre of canvas of 28 x 28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # add gaussian blur to mimic the original MNIST images
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas

def format_for_model(img):
   # Grayscale
   if len(img.shape) >= 3:
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   else:
       gray = img

    # invert colours
   _, inverted = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

   # center and resize the image
   centered_resized = center_and_resize(inverted)

   #formatt
   flat_array = centered_resized.flatten()
   normalized_array = flat_array / 255.0

   final_array = normalized_array.reshape(1, -1)
   return final_array

   