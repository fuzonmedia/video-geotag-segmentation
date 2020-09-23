import cv2
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from calculation import distance_calc11 as dc
import csv
import sys
#old_stdout = sys.stdout
#log_file = open("message.log","w")
#sys.stdout = log_file
print( "this will be written to message.log")
data_log = pd.read_csv("/log_20190715.csv",delimiter=',')

gps_time = data_log["time"]
latitude = data_log["latitude"]
longitude = data_log["longitude"]
time = 0
gps_distance_list = []
gps_distance = 0
gps_distance_new = 0
i=0


gps_distance_main = 0
elapsed_time_in_sec = 0
elapsed_time_in_sec_list = []
lat_vid_list = []
long_vid_list = []
gps_time_list = []


data_log = pd.read_csv("/log_20190715.csv",delimiter=',')
gps_time = data_log["time"]
gps_time_list = data_log["time"].tolist()
latitude = data_log["latitude"]
longitude = data_log["longitude"]
time = 0
gps_distance_list = 0
gps_distance = 0
gps_distance_main = 0

frame_no_list = []

elapsed_time_in_sec_list = []
#intializing csv
with open('pavement_report.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Time,Frame No, Distress,Severity,Latitude,Longitude,Distance '
    writer.writerows([csv_line.split(',')])

cap = cv2.VideoCapture("./VID_20190710_165518.mp4")
cap.set(cv2.CAP_PROP_FPS, 100)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps:",fps)
frame_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
isRecording= True
#print("\n entering in while")
while (cap.isOpened()):
    
    ret,frame = cap.read()
    if ret is True:

        if isRecording == True:
            original_fps = cap.get(cv2.CAP_PROP_FPS) 
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_no_list.append(frame_no)
            elapsed_time_in_sec = int(frame_no/original_fps)
            elapsed_time_in_sec_list.append(elapsed_time_in_sec)
            gps_time_list = data_log["time"].tolist()
            for idx in range(len(gps_time_list)):
                if gps_time_list[idx] == elapsed_time_in_sec:
                   # print("\n time is:",idx)
                    long_vid = longitude[idx]
                    long_vid_list.append(long_vid)
                    lat_vid = latitude[idx]               
                    if idx == 0:
                        pointA = (lat_vid,long_vid)
                        pointB = pointA
                        gps_distance = 0
                        gps_distance_main = 0
                        #print("\nCurrent distance1:",gps_distance)
                       # print("\nTotal distance is1:",gps_distance_main)
                    else:
                        #print("index(idx>0)",idx)
                        lat_vid0 = latitude[idx]
                        long_vid0 = longitude[idx]
                        pointA = (lat_vid0,long_vid0)
                        long_vid1 = longitude[idx+1]
                        lat_vid1 = latitude[idx+1]                    
                        pointB = (lat_vid1,long_vid1) 
                        #print("\n point A is:",pointA, "\npoint B is:",pointB)
                        gps_distance = haversine(pointA,pointB, unit=Unit.METERS)
                        if gps_distance > 0:
                            gps_distance_main = (gps_distance_main +gps_distance)
                        #print("TOTAL DISTANCE(gps main)",(gps_distance_main/fps))                    
                        #print("\nCurrent distance:",gps_distance)                
                        pointA = pointB
                    #gps_distance_main = gps_distance_main/fps
                    speed = round(gps_distance/(elapsed_time_in_sec ),3)
                    cv2.rectangle(frame, (480, 5), (680, 50), (180, 132, 109), -1)
                    cv2.putText(frame,'Total Distance: ' + str(round((gps_distance_main)/fps)) + 'm',(485, 25),font,0.6,(0xFF, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
                    cv2.putText(frame,'latitude: ' + str(lat_vid),(14, 495),font,0.6,(0xFF, 0, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
                    cv2.putText(frame,'longitude: ' + str(long_vid),(14, 530),font,0.6,(0xFF, 0, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
                    cv2.putText(frame,'Speed: ' + str(speed) + 'm/s',(530, 45),font,0.6,(0xFF, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
                    #cv2.putText(frame,': ' + str(speed) + 'm/s',(530, 45),font,0.6,(0xFF, 0, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
            #visualization text to display
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (10, 5), (210, 35), (180, 132, 109), -1)
            cv2.putText(frame,'frame no.: ' + str(frame_no),(15, 25),font, 0.6,(0xFF, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX, )
            cv2.rectangle(frame, (260, 5), (460, 35), (180, 132, 109), -1)
            cv2.putText(frame,'elapsed time.: ' + str(elapsed_time_in_sec) + 's',(265, 25),font, 0.6,(0xFF, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX, )
        
            #cv2.putText(frame,'distance travelled: ' + str(gps_distance),(485, 25),font,0.6,(0, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
            # cv2.putText(frame,'latitude: ' + str(lat_vid),(485, 25),font,0.6,(0, 0xFF, 0xFF),2,cv2.FONT_HERSHEY_SIMPLEX,)
            cv2.imshow('Frame',frame)
            # x- pause and play                     
            if cv2.waitKey(1) & 0xFF == ord('x'): 
                isRecording = False
                print("\npaused")
                #p - pothole
                if cv2.waitKey(0) & 0xFF == ord('p'):
                    print("\n[INFO]Pothole detected by the user")
                    print("\n Latitude",lat_vid)
                    print("\n Longitude",long_vid)
                    print("\n Distance covered",(gps_distance_main/fps))
                    severity = 1                    
                    if cv2.waitKey(0) & 0xFF == ord('1'):
                        print("\n[INFO]minor severity")
                        severity = 1
                    if cv2.waitKey(0) & 0xFF == ord('2'):
                        print("[INFO]medium severity")
                        severity = 2
                    if cv2.waitKey(0) & 0xFF == ord('3'):
                        print("[INFO]major severity")
                        severity = 3
                    with open('pavement_report.csv', 'a') as f:
                        writer = csv.writer(f)
                        detection_line = (str(elapsed_time_in_sec) + "," + str(frame_no) + "," + "Pothole ," +str(severity) 
                                           + "," + str(lat_vid) + "," + str(long_vid) + "," + str(gps_distance_main/fps))
                        writer.writerows([detection_line.split(',')])                       
                    cv2.imwrite("pothole_"+ str(elapsed_time_in_sec) + ".png",frame)

                if cv2.waitKey(0) & 0xFF == ord('x'): 
                    #print("[INFO] video continue")
                    isRecording = True  
                    print("\n playing again")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nYou have pressed q")
                break
      
    #else: 
     #   break

cap.release()
cv2.destroyAllWindows()
#sys.stdout = old_stdout
#log_file.close()

