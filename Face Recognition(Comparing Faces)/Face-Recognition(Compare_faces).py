import face_recognition
import cv2

path1 = 'D:/faces/Astrid.jpg'
path2 = 'D:/faces/Astrid1.jpg'
#path2 = 'D:/faces/Mark Zuckerberg.jpg'

#NOTE : path1 and path2 are the paths of the respective images

Input_face = face_recognition.load_image_file(f'{path1}')
Test_face = face_recognition.load_image_file(f'{path2}')

Input_face = cv2.cvtColor(Input_face,cv2.COLOR_BGR2RGB)
Test_face = cv2.cvtColor(Test_face,cv2.COLOR_BGR2RGB)

faceLoc1 = face_recognition.face_locations(Input_face)[0]
face_encodings1 = face_recognition.face_encodings(Input_face)[0]

faceLoc2 = face_recognition.face_locations(Test_face)[0]
face_encodings2 = face_recognition.face_encodings(Test_face)[0]

cv2.rectangle(Input_face,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(255,255,0),3)
cv2.rectangle(Test_face,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,255,0),3)

result = face_recognition.compare_faces([face_encodings1],face_encodings2)
faceDis = face_recognition.face_distance([face_encodings1],face_encodings2)

Input_face=cv2.resize(Input_face,(800,600))
Test_face=cv2.resize(Test_face,(800,600))

font = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(Test_face,f'{result[0]} {round(faceDis[0],2)}', (50,50), font, 1.0, (0,0, 255), 2)

cv2.imshow('Image1',Input_face)
cv2.imshow('Image2',Test_face)

cv2.waitKey(0)