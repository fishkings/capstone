"""실시간 객체를 이미지로 쪼개주는 코드"""
import cv2 
import time
prev_time = time.time()


cap = cv2.VideoCapture(0)
count = 9000

while cap.isOpened():
    ret, image = cap.read()

    if time.time() - prev_time >= 2:
        resized_image = cv2.resize(image, (192, 192))
        cv2.imwrite(f"C:/imgcollect/{count}.jpg", resized_image)
        print(count)
        count += 1
        prev_time = time.time()
    
    cv2.imshow('Saving Img', image)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        break
   
cap.release()
cv2.destroyAllWindows()


