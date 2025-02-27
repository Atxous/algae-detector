import cv2
import numpy as np
import ntcore

lower = np.array([79, 41, 92])
upper = np.array([94, 255, 255])

inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("jason")
algae_topic = table.getDoubleTopic("algaeXDiff")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open")
        exit()

    while True:   
        ret, frame = cap.read()
        if not ret:
            break
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower, upper)
        noise_reduction = cv2.blur(mask, (12, 12))
        noise_reduction = cv2.inRange(noise_reduction, 1, 80)
        noise_reduction = cv2.blur(noise_reduction, (12, 12))

        circles = cv2.HoughCircles(noise_reduction, 
                                cv2.HOUGH_GRADIENT, 
                                1.3, 
                                minDist = 25, 
                                param1 = 50, 
                                param2 = 70, 
                                minRadius = 10, 
                                maxRadius = 400)

        if circles is not None:
            circles = np.uint16(np.around(circles))[0]
            index = np.argmax(circles[:,2])
            x_pos = circles[index][0]
            y_pos = circles[index][1]

            # Get the pixel difference between the center of the image and the center of the circle
            x_diff = x_pos - frame.shape[1] // 2
            algae_topic.set(x_diff)
