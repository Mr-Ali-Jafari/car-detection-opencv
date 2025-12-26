import cv2
import numpy as np


OBJECT_TRACKERS = {
    'csrt':cv2.legacy.TrackerCSRT_create,
    'mosse': cv2.legacy.TrackerMOSSE_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'boosting': cv2.legacy.TrackerBoosting_create
}

trackers = cv2.legacy.MultiTracker_create()




cap = cv2.VideoCapture("Cars.mp4")


while True:
    frame = cap.read()[1]
    if frame is None:
        break

    frame = cv2.resize(frame,(800,600))



    success , boxes = trackers.update(frame)

    for box in boxes:
        x,y,w,h = [int(c) for c in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)



    cv2.imshow("Tracking",frame)
    k = cv2.waitKey(30)

    if k == ord('s'):
        roi = cv2.selectROI('Tracking',frame)
        tracker = OBJECT_TRACKERS['kcf']()
        trackers.add(tracker,frame,roi)

cap.release()
cv2.destroyAllWindows()
