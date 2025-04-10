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
    
    if debug:
        cv2.imshow("Edges", edges)

    # 2. Parámetros adaptativos para HoughLinesP
    min_line_length = max(warped_image.shape) // 8
    threshold_val = int(min_line_length * 0.7)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold_val,
                            minLineLength=min_line_length, maxLineGap=min_line_length//10)
    if lines is None:
        return None, None

    if debug:
        temp = warped_image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Raw Hough Lines", temp)

    # 3. Clasificar líneas en horizontales y verticales
    horizontals = []
    verticals = []
    height, width = warped_image.shape[:2]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10 or abs(angle) > 170:
            horizontals.append((y1 + y2) / 2.0)
        elif 80 < abs(angle) < 100:
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
        return [np.mean(c) for c in clusters]

    h_lines = cluster_lines(horizontals)
    v_lines = cluster_lines(verticals)

    # 5. Filtrado espacial mejorado: eliminar outliers
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
        return np.linspace(0, height, 9), np.linspace(0, width, 9)

def main():
    # url = "http://192.168.0.105:8080/video"
    url = "http://192.168.1.4:8080/video"
    cap = cv2.VideoCapture(url)

    # Inicializar el diccionario ArUco y el detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Configurar la ventana de salida (única ventana)
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

    print_interval = 3.0
    last_print_time = 0.0

    # Cargar el modelo YOLO (asegúrate de que la ruta es correcta)
    model = YOLO(r"C:\Users\johan\Desktop\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\runs\detect\train6\weights\best.pt")
    # model = YOLO(r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\runs\detect\train7\weights\best.pt")
    # model = YOLO(r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\runs\detetc\train3\weights\best.pt")
    if not cap.isOpened():
        print("No se pudo abrir el stream de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame.")
            break

        # Convertir a escala de grises para detectar ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            # Usar los IDs para fijar la orientación:
            # a8 (top-left): ID=2, h8 (top-right): ID=3,
            # h1 (bottom-right): ID=1, a1 (bottom-left): ID=0
            detected = {}
            for i, marker_id in enumerate(ids):
                detected[marker_id[0]] = np.mean(corners[i][0], axis=0)  # centro del marcador

            if all(k in detected for k in [0, 1, 2, 3]):
                top_left     = detected[2]
                top_right    = detected[3]
                bottom_right = detected[1]
                bottom_left  = detected[0]

                pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                pts_dst = np.array([
                    [0, 0],
                    [800, 0],
                    [800, 800],
                    [0, 800]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame, M, (800, 800))

                # Detección dinámica del grid usando Hough
                h_lines, v_lines = detect_grid_hough(warped, debug=False)

                # Ejecutar inferencia del modelo YOLO sobre la imagen warped
                results = model.predict(warped, conf=0.5)
                annotated = results[0].plot()

                # Dibujar la cuadrícula detectada sobre la imagen anotada
                if h_lines is not None and v_lines is not None:
                    for y in h_lines:
                        cv2.line(annotated, (0, int(y)), (800, int(y)), (0, 255, 0), 2)
                    for x in v_lines:
                        cv2.line(annotated, (int(x), 0), (int(x), 800), (0, 255, 0), 2)

                current_time = time.time()
                if current_time - last_print_time > print_interval:
                    last_print_time = current_time
                    print("Tablero detectado con éxito (Orientación fija por ID).")
                    print(f"  a8 (ID=2): {top_left}")
                    print(f"  h8 (ID=3): {top_right}")
                    print(f"  h1 (ID=1): {bottom_right}")
                    print(f"  a1 (ID=0): {bottom_left}")
                    if h_lines is not None and v_lines is not None:
                        print(f"  Líneas horizontales: {len(h_lines)}")
                        print(f"  Líneas verticales: {len(v_lines)}")
                    print("")

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
