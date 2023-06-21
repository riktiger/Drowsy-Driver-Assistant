import cv2
import sys
import dlib
import numpy as np
from math import hypot
import time
from datetime import datetime
import winsound


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

yawns = 0
blinks = 0
timer=30
time_count=0
blink_status = False
yawn_status = False 
critical_yawn_status=False
critical_blink_status=False
asleep_status=False
alarm_flag=0
abort_status=False
print("DROWSINESS DETECTION AND ALARM MANAGEMENT SYSTEM -version 1.0")
print("PRESS Esc TO QUIT")
print("\n")
time.sleep(1)
print(datetime.now()," :STARTING APPLICATION......")
cap = cv2.VideoCapture(0)
now_start=datetime.now()
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now=datetime.now()
        
    faces = detector(gray)
    for face in faces:
        landmarkseye = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarkseye)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarkseye)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                
                 
        prev_blink_status = blink_status
                
        if blinking_ratio >5 :
           timer -=1
           if timer >20:     
              blink_status=True
              asleep_status=False
              
           elif timer>1:
              blink_status=False
              asleep_status=True
              
                                        
        else:
           blink_status = False 
           if alarm_flag==0:
              timer=30
           
        if prev_blink_status == True and blink_status == False :
           blinks += 1

        if blinks>10:
              critical_blink_status=True
        else :
              critical_blink_status = False
                          

    image_landmarks, lip_distance = mouth_open(frame)
  
    prev_yawn_status = yawn_status    
        
    if lip_distance > 55 :
       yawn_status = True
       if yawns>=6:
          critical_yawn_status=True
       else :
          critical_yawn_status = False
    else:
       yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False :
       yawns += 1
    
    
    if (datetime.now()-now_start).total_seconds()>=60:
       time_count+=1
       print(now," :IN "+str(time_count)+" MIN, THE NUMBER OF BLINKS= "+str(blinks)+" AND THE NUMBER OF YAWNS= "+str(yawns)) 
       if critical_yawn_status==False:
          yawns=0
       if critical_blink_status==False:
          blinks = 0
       now_start=datetime.now()        

    if asleep_status==False:
       if critical_blink_status==True:
          output_text = " NUMBER OF BLINKS THIS PAST MINUTE : " + str(blinks)
          cv2.putText(frame, output_text, (25,400),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
          cv2.putText(frame, "CONDITION : RAPID BLINKING, PROBABLE DROWSINESS", (25,424), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
          cv2.putText(frame, "SOUNDING ALARM, PRESS ENTER TO SILENCE IT", (25,448), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255)) 
          winsound.Beep(400,2000)
          if cv2.waitKey(1)==13:
             print(now," :FOR "+str(blinks)+" BLINKS RAPID BLINKING ALARM WAS SILENCED AND BLINKS RESET TO ZERO")
             blinks=0
             critical_blink_status=False

       else:
          output_text = " NUMBER OF BLINKS THIS PAST MINUTE : " + str(blinks)
          cv2.putText(frame, output_text, (25,400),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
          cv2.putText(frame, "CONDITION : NORMAL", (25,424), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
          
    else:
       alarm_flag=1
       timer-=1
       output_text = " USER IS ASLEEP !!! "
       cv2.putText(frame, output_text, (25,400),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
       cv2.putText(frame, "SOUNDING ALARM ,PRESS ENTER TO SILENCE IT", (25,424), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
       print(now," :WARNING !!! USER IS ASLEEP!!!")
       winsound.Beep(600,1000)
       if cv2.waitKey(1)==13:
          blink_status=True            
          asleep_status=False
          print(now," :ALARM WAS SILENCED")
          alarm_flag=0
       if timer<1:
          print(now," :USER DID NOT WAKE UP")
          print(now," :ABORTING PROGRAM......")
          abort_status=True
          break
          cap.release()
          cv2.destroyAllWindows()

    if critical_yawn_status == True:
        output_text = " NUMBER OF YAWNS THIS PAST MINUTE : " + str(yawns)
        cv2.putText(frame, output_text, (25,12),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
        cv2.putText(frame, "CONDITION : HIGHLY EXHAUSTED, SOUNDING ALARM, PRESS ENTER TO SILENCE IT", (25,36), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
        cv2.putText(frame, "SOUNDING ALARM, PRESS ENTER TO SILENCE IT", (25,60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,0,255))
        winsound.Beep(200,2000)
        if cv2.waitKey(1)==13:
           print(now," :FOR "+str(yawns)+" YAWNS EXCESSIVE YAWNING ALARM WAS SILENCED AND YAWNS RESET TO ZERO")
           yawns=0
           critical_yawn_status=False
    else:
       output_text = " NUMBER OF YAWNS THIS PAST MINUTE : " + str(yawns)
       cv2.putText(frame, output_text, (25,12),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
       cv2.putText(frame, "CONDITION : NORMAL", (25,36), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))

           
    cv2.imshow('VIEWFINDER', frame )
            
    if cv2.waitKey(1) == 27: 
       break

print("\n")
print(datetime.now()," :Esc WAS PRESSED")
print(datetime.now()," :EXITING......")     
cap.release()
cv2.destroyAllWindows()