import cv2
import numpy as np
import time

def main():
    # url = "http://192.168.0.105:8080/video"
    url = "http://192.168.1.4:8080/video"
    # url = "http://10.209.154.25:8080/video"
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

        if ids is not None and len(ids) == 4:
            # Crear diccionario para mapear id -> esquinas
            detected = {}
            for i, marker_id in enumerate(ids):
                detected[marker_id[0]] = corners[i][0]

            if all(k in detected for k in [0,1,2,3]):
                # Función para obtener el centro (promedio) de las 4 esquinas
                def marker_center(pts):
                    return (np.mean(pts[:,0]), np.mean(pts[:,1]))

                # Según tu configuración:
                # a8: ID=2  → top-left
                # h8: ID=3  → top-right
                # h1: ID=1  → bottom-right
                # a1: ID=0  → bottom-left
                top_left     = marker_center(detected[2])
                top_right    = marker_center(detected[3])
                bottom_right = marker_center(detected[1])
                bottom_left  = marker_center(detected[0])

                pts_src = np.array([
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left
                ], dtype=np.float32)

                pts_dst = np.array([
                    [0, 0],
                    [800, 0],
                    [800, 800],
                    [0, 800],
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame, M, (800, 800))

                # Dibujar cuadrícula 8x8 (cada celda de 100x100)
                for row in range(9):
                    y = row * 100
                    cv2.line(warped, (0, y), (800, y), (0,255,0), 2)
                for col in range(9):
                    x = col * 100
                    cv2.line(warped, (x, 0), (x, 800), (0,255,0), 2)

                # Imprimir mensaje resumido cada print_interval segundos
                current_time = time.time()
                if current_time - last_print_time > print_interval:
                    last_print_time = current_time
                    print("Tablero detectado con éxito.")
                    print(f"  a8 (ID=2): ({top_left[0]:.2f}, {top_left[1]:.2f})")
                    print(f"  h8 (ID=3): ({top_right[0]:.2f}, {top_right[1]:.2f})")
                    print(f"  h1 (ID=1): ({bottom_right[0]:.2f}, {bottom_right[1]:.2f})")
                    print(f"  a1 (ID=0): ({bottom_left[0]:.2f}, {bottom_left[1]:.2f})")
                    print("")

                cv2.imshow("Tablero Amazons", warped)

        # Se omite mostrar la ventana original
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
