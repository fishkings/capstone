"""동영상을 프레임 단위로 쪼개주는 코드"""
import cv2

vidcap = cv2.VideoCapture(r'C:\Users\kcjer\OneDrive\바탕 화면\1.mp4')

count = 6000

while (vidcap.isOpened()):
    ret, image = vidcap.read()

    if (int(vidcap.get(1)) % 6000 == 0):
        resized_image = cv2.resize(image, (192, 192))
        cv2.imwrite(f"C:/imgcollect/{count}.jpg", resized_image)
        print(count)
        count += 1

vidcap.release()