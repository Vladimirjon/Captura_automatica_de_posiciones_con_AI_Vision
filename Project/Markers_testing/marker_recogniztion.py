import cv2

def main():
    # URL del stream de la cámara (IP Webcam u otra)
    url = "http://192.168.0.105:8080/video"
    cap = cv2.VideoCapture(url)
    
    # En versiones actuales, se recomienda usar getPredefinedDictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Parámetros del detector
    aruco_params = cv2.aruco.DetectorParameters()
    # Create detector instance
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    if not cap.isOpened():
        print("No se pudo abrir el flujo de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame del stream.")
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar marcadores usando el detector
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Si se detectan marcadores
        if ids is not None and len(ids) > 0:
            # Dibujar contornos e IDs
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print("Marcadores detectados con IDs:", ids.flatten())

        # Mostrar en ventana
        cv2.imshow("Deteccion ArUco", frame)
        
        # Salir con la tecla ESC (código 27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if cv2.getWindowProperty("Deteccion ArUco", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
