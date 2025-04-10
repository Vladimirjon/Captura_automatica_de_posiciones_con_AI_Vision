import cv2
import numpy as np
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox

import torch

# Guarda la función original de torch.load
original_torch_load = torch.load

# Define una versión parcheada que fuerce weights_only=False
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Reemplaza torch.load con la versión parcheada
torch.load = patched_torch_load


def main():
    # Configuración de la cámara (stream IP)
    url = "http://192.168.0.105:8080/video"
    cap = cv2.VideoCapture(url)

    # Cargar el modelo YOLOv8 entrenado (asegúrate de que "best.pt" está en la ruta correcta)
    model = YOLO(r"C:\Users\johan\Desktop\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\runs\detect\train6\weights\best.pt")

    # Configurar el detector de marcadores ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Configurar la ventana de salida (redimensionable)
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)

    # Configurar el VideoWriter para grabar el video (800x800, 20 fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (800, 800))

    # Intervalo de impresión para mensajes (3 segundos)
    print_interval = 3.0
    last_print_time = 0.0

    # Variable para almacenar el conteo final (usaremos el último frame con detecciones)
    final_counts = None

    if not cap.isOpened():
        print("No se pudo abrir el stream de video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame.")
            break

        # Convertir a escala de grises y detectar marcadores
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) == 4:
            # Crear un diccionario para mapear cada ID a sus esquinas
            detected = {}
            for i, marker_id in enumerate(ids):
                detected[marker_id[0]] = corners[i][0]

            if all(k in detected for k in [0, 1, 2, 3]):
                # Función para calcular el centro de cada marcador
                def marker_center(pts):
                    return (np.mean(pts[:, 0]), np.mean(pts[:, 1]))

                # Asignación de acuerdo a la configuración:
                # a8: ID=2 → top-left
                # h8: ID=3 → top-right
                # h1: ID=1 → bottom-right
                # a1: ID=0 → bottom-left
                top_left     = marker_center(detected[2])
                top_right    = marker_center(detected[3])
                bottom_right = marker_center(detected[1])
                bottom_left  = marker_center(detected[0])

                # Definir puntos fuente (de la imagen original) y destino (tablero plano 800×800)
                pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                pts_dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)

                # Calcular la transformación y aplicar warpPerspective
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame, M, (800, 800))

                # Dibujar la cuadrícula 8×8 (cada celda 100×100)
                for row in range(9):
                    y = row * 100
                    cv2.line(warped, (0, y), (800, y), (0, 255, 0), 2)
                for col in range(9):
                    x = col * 100
                    cv2.line(warped, (x, 0), (x, 800), (0, 255, 0), 2)

                # Ejecutar inferencia sobre la imagen warpeada
                results = model.predict(warped, conf=0.5)
                annotated = results[0].plot()  # Genera la imagen anotada con detecciones

                # Contar las detecciones de cada clase (suponiendo que results[0].boxes.cls contiene índices de clase)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    cls_array = results[0].boxes.cls.cpu().numpy().astype(int)
                    unique, counts = np.unique(cls_array, return_counts=True)
                    final_counts = dict(zip(unique, counts))
                else:
                    final_counts = {}

                # Imprimir mensaje resumido cada 3 segundos
                current_time = time.time()
                if current_time - last_print_time > print_interval:
                    last_print_time = current_time
                    print("Tablero detectado con éxito.")
                    print(f"  a8 (ID=2): ({top_left[0]:.2f}, {top_left[1]:.2f})")
                    print(f"  h8 (ID=3): ({top_right[0]:.2f}, {top_right[1]:.2f})")
                    print(f"  h1 (ID=1): ({bottom_right[0]:.2f}, {bottom_right[1]:.2f})")
                    print(f"  a1 (ID=0): ({bottom_left[0]:.2f}, {bottom_left[1]:.2f})")
                    print("")

                # Mostrar la imagen anotada y grabarla en el video
                cv2.imshow("Tablero Amazons", annotated)
                out.write(annotated)

        # Manejo de teclas y ventana
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break
        if cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Mostrar un resumen final en una ventana emergente (popup)
    # Mapear índices a nombres (asumiendo: 0: arrow, 1: black_amazon, 2: white_amazon)
    class_names = {0: "arrow", 1: "black_amazon", 2: "white_amazon"}
    summary = "Final Detection Counts:\n"
    summary += f"Arrow: {final_counts.get(0, 0)}\n"
    summary += f"Black Amazon: {final_counts.get(1, 0)}\n"
    summary += f"White Amazon: {final_counts.get(2, 0)}"

    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    messagebox.showinfo("Detection Summary", summary)
    root.destroy()

if __name__ == "__main__":
    main()
