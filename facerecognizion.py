import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone


# url = "http://192.168.1.7:4747/video"
#
# cap = cv2.VideoCapture(url)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Resources/bg2.png")

if imgBackground is None:
    print("Error: Background image not loaded. Check the file path.")
    exit()

folderModePath = "Resources/mode"
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgMode = cv2.imread(os.path.join(folderModePath, path))
    if imgMode is not None:
        imgModeList.append(imgMode)
    else:
        print(f"Warning: Image {path} not loaded. Check the file.")

# print(len(imgModeList))

# Resize images in imgModeList to (414, 633)
imgModeList_resized = [cv2.resize(img, (414, 633)) for img in imgModeList]

# Load encodings and student IDs
print("Loading encodings...")
try:
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
        if isinstance(encodeListKnownWithIds, tuple) and len(encodeListKnownWithIds) == 2:
            encodeListKnown, studentIds = encodeListKnownWithIds
            print(studentIds)
        else:
            print("Error: The structure of the data in EncodeFile.p is not as expected.")
            exit()
except Exception as e:
    print(f"Error: {e}")
    exit()
print("Loading encodings...Done")


while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from camera.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)


    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList_resized[1]

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = None
        if len(faceDis) > 0:
            matchIndex = faceDis.argmin()
            # print(matchIndex)
        if matches[matchIndex]:
            name = studentIds[matchIndex]
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(imgBackground, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imgBackground, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgBackground, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(imgBackground, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(imgBackground, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgBackground, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # print('Matchs ', matches)
        # print('Face Dis ', faceDis)

        matchIndex = np.argmin(faceDis)
        # print('Match Index ', matchIndex)

        if matches [matchIndex]:
            print("Know Face Detected")
            print(studentIds[matchIndex])



    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
