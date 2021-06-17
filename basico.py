import cv2
import numpy as np
import face_recognition

imgLuiza = face_recognition.load_image_file('ImagensBasicas/luiza1.png')
imgLuiza = cv2.cvtColor(imgLuiza.cv2.COLOR_BGR2RGB)
imgTeste = face_recognition.load_image_file('ImagensBasicas/luiza2.png')
imgTeste = cv2.cvtColor(imgTeste.cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgLuiza)[0]
encodeLuiza = face_recognition.face_encodings(imgLuiza)[0]
cv2.rectangle(imgLuiza,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)

faceLocTeste = face_recognition.face_locations(imgTeste)[0]
encodeTeste = face_recognition.face_encodings(imgTeste)[0]
cv2.rectangle(imgTeste,(faceLocTeste[3], faceLocTeste[0]),(faceLocTeste[1], faceLocTeste[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeLuiza], encodeTeste)
faceDis = face_recognition.face_distance([encodeLuiza], encodeTeste)
print(results)

cv2.putText(imgTeste, f'{results}{round(faceDis[0 ],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)


cv2.imshow('Luiza 1', imgLuiza)
cv2.imshow('Luiza 2'), imgTeste)
cv2.waitKey(0)


