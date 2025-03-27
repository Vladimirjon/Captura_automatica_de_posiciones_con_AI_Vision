import cv2
import numpy as np
import os


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

output_folder = r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Project\Markers"
os.makedirs(output_folder, exist_ok=True)

for marker_id in range(4):
    marker_image = cv2.aruco.drawMarker(aruco_ditc, marker_id, 200)
    cv2.imwrite(os.path.join(output_folder, f"marker_{marker_id}.png"), marker_image)
    
print("Markers generated successfully")