import cv2
import mediapipe  as mp
import time

# Human pose detection.
# https://google.github.io/mediapipe/solutions/pose
# https://google.github.io/mediapipe/

# Human pose classification..
# https://colab.research.google.com/drive/1z4IM8kG6ipHN6keadjD-F6vMiIIgViKK#scrollTo=fgHTsKdz7cn_

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,
                                     self.detectionCon,self.trackCon)

    def findpose(self,img,draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return (img)

    def findPosition(self,img,draw=True):
        lmList = []
        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])

                if draw:
                    cv2.circle(img,(cx,cy), 5,(0,0,255),cv2.FILLED)
        return lmList


#
def main():
    cap = cv2.VideoCapture("/home/vert/iCamPlus/Video.mp4")
    ptime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findpose(img)
        lmlist = detector.findPosition(img,draw=False)
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0, 0, 0), cv2.FILLED)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0))

        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()