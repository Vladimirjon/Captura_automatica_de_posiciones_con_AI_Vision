import cv2
import numpy as np
import time

def main():
    url = "http://192.168.0.105:8080/video"
    cap = cv2.VideoCapture(url)

    # Diccionario ArUco y detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Configurar ventana de salida para que sea redimensionable
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

    # Intervalo de impresión (en segundos)
    print_interval = 3.0
    last_print_time = 0.0

    if not cap.isOpened():
        print("No se pudo abrir el stream de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Verificar que se detectaron 4 marcadores
        if ids is not None and len(ids) == 4:
            # Función para obtener el centro (promedio) de las 4 esquinas
            def marker_center(pts):
                return (np.mean(pts[:, 0]), np.mean(pts[:, 1]))
            
            # Calcular el centro de cada marcador y guardarlo en una lista
            points = []
            for i in range(len(ids)):
                pts = corners[i][0]
                c = marker_center(pts)
                points.append(c)
            
            # Convertir la lista a un array para facilitar cálculos
            points = np.array(points)

            # Calcular la suma y la diferencia de cada punto
            s = points.sum(axis=1)    # x + y
            diff = points[:, 0] - points[:, 1]  # x - y

            # Ordenar los puntos:
            # Top-left: suma mínima
            # Bottom-right: suma máxima
            # Top-right: diferencia mínima
            # Bottom-left: diferencia máxima
            top_left = points[np.argmin(s)]
            bottom_right = points[np.argmax(s)]
            top_right = points[np.argmin(diff)]
            bottom_left = points[np.argmax(diff)]

            pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            pts_dst = np.array([
                [0, 0],
                [800, 0],
                [800, 800],
                [0, 800]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(frame, M, (800, 800))

            # Dibujar la cuadrícula 8x8 (cada celda de 100x100)
            for row in range(9):
                y = row * 100
                cv2.line(warped, (0, y), (800, y), (0, 255, 0), 2)
            for col in range(9):
                x = col * 100
                cv2.line(warped, (x, 0), (x, 800), (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_print_time > print_interval:
                last_print_time = current_time
                print("Tablero detectado con éxito.")
                print(f"  Top-left: ({top_left[0]:.2f}, {top_left[1]:.2f})")
                print(f"  Top-right: ({top_right[0]:.2f}, {top_right[1]:.2f})")
                print(f"  Bottom-right: ({bottom_right[0]:.2f}, {bottom_right[1]:.2f})")
                print(f"  Bottom-left: ({bottom_left[0]:.2f}, {bottom_left[1]:.2f})")
                print("")

            cv2.imshow("Tablero Amazons", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
