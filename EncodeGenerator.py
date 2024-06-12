import cv2
import face_recognition
import pickle
import os

folderPath = 'images/resized'
pathlist = os.listdir(folderPath)
print(pathlist)
imgList = []
studentIds = []

for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encode = encodings[0]
            encodeList.append(encode)
        else:
            print("Warning: No face found in image. Skipping.")
    return encodeList

print('Encoding known faces...')
encodeListKnown = findEncodings(imgList)

print('Encoding known faces...Done')

# Save the encodings and corresponding student IDs
encodeListKnownWithIds = (encodeListKnown, studentIds)
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)
print('Encoding saved to file...Done')
