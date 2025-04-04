import cv2

url = "http://192.168.0.105:8080/video"
cap = cv2.VideoCapture(url)

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Stream", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if cv2.getWindowProperty("Stream", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
