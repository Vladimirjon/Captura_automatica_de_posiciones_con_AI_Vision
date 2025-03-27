import cv2
import numpy as np

# IP del servidor IP WebCam del celular 
url = "http://192.168.0.105:8080/video"
cap = cv2.VideoCapture(url)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stream", 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Verifica si hay 4 marcadores
    if ids is not None and len(ids) == 4:
        detected_ids = [id_[0] for id_ in ids] 

        if all(x in detected_ids for x in [0,1,2,3]):
            print("Tablero reconocido !")

            cv2.putText(frame,
                        "Board detected !",            # Texto
                        (50,50),                      # Coordenadas donde dibujar el texto
                        cv2.FONT_HERSHEY_SIMPLEX,     # Fuente
                        1.0,                          # Escala de fuente
                        (0,255,0),                    # Color (B,G,R) → Verde
                        2,                            # Grosor de línea
                        cv2.LINE_AA)


    cv2.imshow("Stream", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    # Cerrar la ventana 
    if cv2.getWindowProperty("Stream", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
