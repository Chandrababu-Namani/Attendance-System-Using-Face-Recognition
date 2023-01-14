import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
path_for_Images = 'D:/faces/' #Give your respective directory path here
faces = []
face_names = []
cur_dir = os.listdir(path_for_Images)
for face in cur_dir:
    img = cv2.imread(f'{path_for_Images}/{face}')
    faces.append(img)
    face_names.append(face[:-4])

def finding_encodings(faces):
    encodings = []
    for face in faces:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(face)[0]
        encodings.append(encode)
    return encodings

faces_encodings = finding_encodings(faces)
print('Encodings found for faces.')

def mark_Attendance(name):
    with open('Attendance.csv','r+') as f: #'Attendence.csv' is a csv file in the same directory with text('Name,Time') in it
        Data = f.readlines()
        name_list = []
        for line in Data:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            cur_datetime = datetime.now()
            date_time = cur_datetime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_time}')

process_this_frame = True
capture = cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25) #Minizing the frame to 1/4th
    small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame,face_locations)
    names=[]
    
    if process_this_frame:
        for encode,faceLoc in zip(face_encodings,face_locations):
            
            matches = face_recognition.compare_faces(faces_encodings,encode)
            face_distances = face_recognition.face_distance(faces_encodings,encode)

            best_match_Index = np.argmin(face_distances)
            name = 'Unknown'
            
            if matches[best_match_Index]:
                name = face_names[best_match_Index]
                mark_Attendance(name)
            names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 4)

        # Input text label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    frame = cv2.resize(frame,(800,600))
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break