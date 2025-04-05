import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ----- Patch para evitar el error de weights_only -----
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
# --------------------------------------------------------

def detect_grid_hough(warped_image, debug=False):
    """
    Detecta dinámicamente las líneas del tablero usando la transformada de Hough.
    Retorna dos listas (h_lines, v_lines) con las posiciones (en píxeles)
    de las líneas horizontales y verticales detectadas.
    Si no se detectan 9 líneas en cada dirección, se retorna un grid equidistante.
    """
    # 1. Preprocesamiento
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 2. Parámetros adaptativos para HoughLinesP
    min_line_length = max(warped_image.shape) // 8
    threshold_val = int(min_line_length * 0.7)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold_val,
                            minLineLength=min_line_length, maxLineGap=min_line_length//10)
    if lines is None:
        return None, None

    # 3. Clasificar líneas en horizontales y verticales
    horizontals = []
    verticals = []
    height, width = warped_image.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        # Línea horizontal (cerca de 0° o 180°)
        if abs(angle) < 10 or abs(angle) > 170:
            horizontals.append((y1 + y2) / 2.0)
        # Línea vertical (cerca de 90°)
        elif 80 < abs(angle) < 100:
            verticals.append((x1 + x2) / 2.0)

    # 4. Agrupar líneas similares (clustering)
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
        return [np.mean(c) for c in clusters]

    h_lines = cluster_lines(horizontals)
    v_lines = cluster_lines(verticals)

    # 5. Filtrado de outliers
    def filter_outliers(lines, expected_count=9):
        if len(lines) <= expected_count:
            return lines
        sorted_lines = sorted(lines)
        spacings = np.diff(sorted_lines)
        median_spacing = np.median(spacings)
        filtered = [sorted_lines[0]]
        for i in range(1, len(sorted_lines)):
            if abs(sorted_lines[i] - sorted_lines[i-1] - median_spacing) < median_spacing * 0.3:
                filtered.append(sorted_lines[i])
        return filtered

    h_lines = filter_outliers(h_lines)
    v_lines = filter_outliers(v_lines)

    # 6. Verificar si tenemos 9 líneas en cada dirección
    if len(h_lines) == 9 and len(v_lines) == 9:
        return sorted(h_lines), sorted(v_lines)
    else:
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

    # Solo una ventana
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

    print_interval = 3.0
    last_print_time = 0.0

    # Cargar modelo YOLO
    model = YOLO(r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\runs\detect\train6\weights\best.pt")

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
            # Ordenar marcadores dinámicamente
            def marker_center(pts):
                return (np.mean(pts[:, 0]), np.mean(pts[:, 1]))

            points = []
            for i in range(len(ids)):
                pts = corners[i][0]
                points.append(marker_center(pts))
            points = np.array(points)

            s = points.sum(axis=1)
            diff = points[:, 0] - points[:, 1]
            top_left = points[np.argmin(s)]
            bottom_right = points[np.argmax(s)]
            top_right = points[np.argmin(diff)]
            bottom_left = points[np.argmax(diff)]

            # Warp a 800x800
            pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            pts_dst = np.array([
                [0, 0],
                [800, 0],
                [800, 800],
                [0, 800]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(frame, M, (800, 800))

            # Detectar líneas con Hough
            h_lines, v_lines = detect_grid_hough(warped, debug=False)

            # Inferencia de YOLO sobre la imagen warped (sin líneas)
            results = model.predict(warped, conf=0.5)
            annotated = results[0].plot()  # Imagen con bounding boxes y etiquetas

            # Dibujar la cuadrícula en la imagen YA anotada por YOLO
            if h_lines is not None and v_lines is not None:
                for y in h_lines:
                    cv2.line(annotated, (0, int(y)), (800, int(y)), (0, 255, 0), 2)
                for x in v_lines:
                    cv2.line(annotated, (int(x), 0), (int(x), 800), (0, 255, 0), 2)

            # Mensajes de debug cada 3 seg
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

            # Mostrar todo en una sola ventana
            cv2.imshow("Tablero Amazons", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
