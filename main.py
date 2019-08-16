import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret, frame = cap.read()

    # cv2.imshow('frame', frame)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted([ (cv2.contourArea(i), i) for i in contours ], key=lambda a:a[0], reverse=True)

    for _, contour in sorted_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
   

    _, card_contour = sorted_contours[1]
    cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 2)

    



    rect = cv2.minAreaRect(card_contour)
    points = cv2.boxPoints(rect)
    points = np.int0(points)

    for point in points:
        cv2.circle(frame, tuple(point), 10, (0,255,0), -1)


    # create a min area rectangle from our contour
    _rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(_rect)
    box = np.int0(box)

    # create empty initialized rectangle
    rect = np.zeros((4, 2), dtype = "float32")

    # get top left and bottom right points
    s = box.sum(axis = 1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]

    # get top right and bottom left points
    diff = np.diff(box, axis = 1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    cv2.imshow('draw contours',frame)

    # rect = cv2.minAreaRect(card_contour)
    # points = cv2.boxPoints(rect)
    # points = np.int0(points)

    # for point in points:
    #     cv2.circle(frame, tuple(point), 10, (0,255,0), -1)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# from datetime import datetime
  
# class Webcam(object):
  
#     WINDOW_NAME = "Playing Card Detection System"
  
#     # constructor
#     def __init__(self):
#         self.webcam = cv2.VideoCapture(0)       
 
#     # save image to disk
#     def _save_image(self, path, image):
#         filename = "<span class="skimlinks-unlinked">datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f</span>" '.jpg'
#         cv2.imwrite(path + filename, image)
  
#     # detect cards in webcam
#     def detect_cards(self, cascade_path):
  
#         # get image from webcam
#         img = <span class="skimlinks-unlinked">self.webcam.read</span>()[1]
          
#         # do card detection
#         card_cascade = cv2.CascadeClassifier(cascade_path)
  
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         cards = card_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(200, 300))
  
#         for (x,y,w,h) in cards:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  
#         # save image to disk
#         self._save_image('WebCam/Detection/', img)
  
#         # show image in window
#         cv2.imshow(self.WINDOW_NAME, img)
#         cv2.waitKey(2000)
#         cv2.destroyAllWindows()
          
#         # indicate whether cards detected
#         if len(cards) > 0:
#             return True
  
#         return False




 
# webcam = Webcam()
  
# # play a game of cards
# while True:
  
#     # attempt to detect the three of hearts
#     if webcam.detect_cards('haarcascade_3hearts.xml') == True:
#         print "I have the cotton picking three of hearts"
#     else:
#         print "I do not have the darn gun slinging three of hearts"
