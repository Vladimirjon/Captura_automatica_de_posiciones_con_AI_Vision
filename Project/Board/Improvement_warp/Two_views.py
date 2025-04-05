import cv2
import numpy as np
import time

def detect_grid_hough(warped_image):
    """
    Detecta dinámicamente las líneas del tablero usando la transformada de Hough.
    Retorna dos listas (h_lines, v_lines) con las posiciones (en pixeles) 
    de las líneas horizontales y verticales detectadas.
    Si no se detectan 9 líneas en cada dirección, se retorna un grid equidistante.
    """
    # 1. Preprocesamiento
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 2. Detección de líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return None, None

    # 3. Clasificar líneas en horizontales/verticales
    horizontals = []
    verticals = []
    height, width = warped_image.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

        # Línea horizontal (ángulo cercano a 0° o 180°)
        if abs(angle) < 10 or abs(angle) > 170:
            # Tomamos el promedio de y1, y2 para representar su "posición"
            horizontals.append((y1 + y2) / 2.0)

        # Línea vertical (ángulo cercano a 90°)
        elif 80 < abs(angle) < 100:
            # Tomamos el promedio de x1, x2 para representar su "posición"
            verticals.append((x1 + x2) / 2.0)

    # 4. Agrupar líneas similares (clustering sencillo)
    def cluster_lines(positions, threshold=20):
        if not positions:
            return []
        positions = sorted(positions)
        clusters = [[positions[0]]]

        for pos in positions[1:]:
            if abs(pos - clusters[-1][-1]) < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        # Retorna la media de cada cluster
        return [np.mean(c) for c in clusters]

    h_lines = cluster_lines(horizontals)
    v_lines = cluster_lines(verticals)

    # 5. Verificar si tenemos un tablero 8x8 (9 líneas horizontales y 9 verticales)
    #    De lo contrario, fallback a un grid equidistante
    if len(h_lines) == 9 and len(v_lines) == 9:
        return sorted(h_lines), sorted(v_lines)
    else:
        # Fallback: generar un grid equidistante basado en (width, height)
        return (
            np.linspace(0, height, 9),
            np.linspace(0, width, 9)
        )

def main():
    url = "http://192.168.0.105:8080/video"
    cap = cv2.VideoCapture(url)

    # Diccionario ArUco y detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Ventanas
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

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

        # Mostrar la imagen original
        cv2.imshow("Original", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            # Función para obtener el centro de cada marcador
            def marker_center(pts):
                return (np.mean(pts[:, 0]), np.mean(pts[:, 1]))

            # Calcular el centro de cada marcador
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

            # Definir un warp de 800x800 (fijo, como tu código original)
            pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            pts_dst = np.array([
                [0, 0],
                [800, 0],
                [800, 800],
                [0, 800]
            ], dtype=np.float32)

            # Matriz de transformación y warp
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(frame, M, (800, 800))

            # Detectar grid dinámico con Hough
            h_lines, v_lines = detect_grid_hough(warped)

            # Dibujar las líneas resultantes
            warped_with_grid = warped.copy()
            if h_lines is not None and v_lines is not None:
                # Dibujar líneas horizontales
                for y in h_lines:
                    y_int = int(y)
                    cv2.line(warped_with_grid, (0, y_int), (800, y_int), (0, 255, 0), 2)
                # Dibujar líneas verticales
                for x in v_lines:
                    x_int = int(x)
                    cv2.line(warped_with_grid, (x_int, 0), (x_int, 800), (0, 255, 0), 2)

            # Mensajes de debug
            current_time = time.time()
            if current_time - last_print_time > print_interval:
                last_print_time = current_time
                print("Tablero detectado con éxito.")
                print(f"  top_left: {top_left}")
                print(f"  top_right: {top_right}")
                print(f"  bottom_right: {bottom_right}")
                print(f"  bottom_left: {bottom_left}")
                if h_lines is not None and v_lines is not None:
                    print(f"  Líneas horizontales detectadas: {len(h_lines)}")
                    print(f"  Líneas verticales detectadas: {len(v_lines)}")
                else:
                    print("  No se detectó ninguna línea con Hough.")
                print("")

            cv2.imshow("Tablero Amazons", warped_with_grid)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
