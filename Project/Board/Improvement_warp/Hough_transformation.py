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

    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

    # Número de filas y columnas del tablero
    num_rows = 8
    num_cols = 8

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
            # Calcular el centro de cada marcador
            def marker_center(pts):
                return (np.mean(pts[:, 0]), np.mean(pts[:, 1]))

            points = []
            for i in range(len(ids)):
                pts = corners[i][0]
                c = marker_center(pts)
                points.append(c)
            points = np.array(points)

            # Ordenar los 4 puntos automáticamente
            s = points.sum(axis=1)    
            diff = points[:, 0] - points[:, 1]
            top_left = points[np.argmin(s)]
            bottom_right = points[np.argmax(s)]
            top_right = points[np.argmin(diff)]
            bottom_left = points[np.argmax(diff)]

            # Calcular el ancho y alto en la imagen original
            board_width = int(np.linalg.norm(top_right - top_left))
            board_height = int(np.linalg.norm(bottom_left - top_left))

            # Definir los puntos destino en base al ancho y alto dinámicos
            pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            pts_dst = np.array([
                [0, 0],
                [board_width, 0],
                [board_width, board_height],
                [0, board_height]
            ], dtype=np.float32)

            # Obtener la matriz de transformación
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)

            # Aplicar la transformación con el tamaño dinámico
            warped = cv2.warpPerspective(frame, M, (board_width, board_height))

            # Calcular el tamaño de cada celda
            cell_width = board_width / num_cols
            cell_height = board_height / num_rows

            # Dibujar la cuadrícula
            for row in range(num_rows + 1):
                y = int(row * cell_height)
                cv2.line(warped, (0, y), (board_width, y), (0, 255, 0), 2)
            for col in range(num_cols + 1):
                x = int(col * cell_width)
                cv2.line(warped, (x, 0), (x, board_height), (0, 255, 0), 2)

            # Mensajes de debug
            current_time = time.time()
            if current_time - last_print_time > print_interval:
                last_print_time = current_time
                print("Tablero detectado con éxito.")
                print(f"  top_left: {top_left}")
                print(f"  top_right: {top_right}")
                print(f"  bottom_right: {bottom_right}")
                print(f"  bottom_left: {bottom_left}")
                print(f"  board_width={board_width}, board_height={board_height}")
                print(f"  cell_width={cell_width:.2f}, cell_height={cell_height:.2f}\n")

            cv2.imshow("Tablero Amazons", warped)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
