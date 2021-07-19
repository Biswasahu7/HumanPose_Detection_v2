import cv2
import mediapipe  as mp
import time


# refer
# https://google.github.io/mediapipe/solutions/pose
# https://google.github.io/mediapipe/

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("/home/vert/iCamPlus/iCamPlus.mp4")
ptime = 0
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img,(cx,cy), 5,(0,0,255),cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (70,50),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0))

    cv2.imshow("Video", img)
    cv2.waitKey(10)
