import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture("/home/vert/iCamPlus/iCamPlus.mp4")
# cap = cv2.VideoCapture("/home/vert/iCamPlus/Video.mp4")
ptime=0
save_image=0
Detection=0

# Calling posemodule from the folder to detect
detector = pm.poseDetector()


# Running while loop into the image to get get the human body point
while True:

    # Reading image from video
    success, img = cap.read()

    # Here we can find the pose of a human body through the help of d
    img = detector.findpose(img)

    # RESIZING image FRAME to display...
    scale_percent = 120
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Resize original image to display according to our requirement...
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    height, width, _ = img.shape
    size = (width, height)

    #
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) == 0:
        continue
        # print("Person is not available")
    # print(len(lmlist))
    # print(lmlist[24])
    cv2.circle(img, (lmlist[24][1], lmlist[24][2]), 8, (0, 0, 0), cv2.FILLED)
    cv2.circle(img, (lmlist[15][1], lmlist[15][2]), 8, (0, 0, 0), cv2.FILLED)
    # cv2.circle(img, (lmlist[28][1], lmlist[28][2]), 8, (0, 0, 0), cv2.FILLED)

    # Right_hip location
    x = (lmlist[24][2])

    # Left_wrist location
    y = (lmlist[15][2])
    z = (lmlist[28][2])
    # print("Right_hip_{}".format(x))
    # print("Right_shoulder-{}".format(y))

    # Taking distance from Right_hip to Left_Wrist.
    val=x-y

   # checking condition for alret message.
    if val > 150:
        Detection +=1
        # cv2.imwrite("/home/vert/iCamPlus/Crop_Image/image_{}.jpg".format(save_image), img)
        save_image += 1
        # print(val)
        if Detection == 1:
            print("Position is detected we need give alert")
            cv2.imwrite("/home/vert/iCamPlus/Crop_Image/image_{}.jpg".format(save_image), img)
        # cv2.imwrite("/media/jsw/Data/Crop_Image/image_{}.jpg".format(save_image), img_crop)

    # ctime = time.time()
    # fps = 1 / (ctime - ptime)
    # ptime = ctime
    # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0))

    cv2.imshow("Video", img)
    cv2.waitKey(1)


