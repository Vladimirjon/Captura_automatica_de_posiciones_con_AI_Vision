import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import io
import contextlib
import os 


def detect_move(last_board, current_board):
    moved_from = None
    moved_to = None
    arrow_added = None

    for row in range(8):
        for col in range(8):
            old_val = last_board[row][col]
            new_val = current_board[row][col]

            if old_val != new_val:
                if old_val in ['W', 'B'] and new_val == '0':
                    moved_from = (row, col, old_val)

                if old_val == '0' and new_val in ['W', 'B']:
                    moved_to = (row, col, new_val)

                if old_val == '0' and new_val == 'A':
                    arrow_added = (row, col)

    def to_notation(row, col):
        return f"{chr(ord('a') + col)}{8 - row}"

    if moved_from and moved_to:
        piece_type = 'W' if moved_from[2] == 'W' else 'B'
        from_square = to_notation(moved_from[0], moved_from[1])
        to_square = to_notation(moved_to[0], moved_to[1])
        arrow_str = ""
        if arrow_added:
            arrow_str = f", A: {to_notation(arrow_added[0], arrow_added[1])}"
        return f"{piece_type}: {from_square} -> {to_square}{arrow_str}"

    return ""

# ----- Patch para evitar el error de weights_only -----
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
# --------------------------------------------------------

# ==================== Clases para seguimiento del tablero ====================
@dataclass
class Piece:
    type: str          # 'amazona' o 'flecha'
    color: str         # 'white' o 'black'
    position: Tuple[int, int]  # (fila, columna)

class BoardTracker:
    def __init__(self, grid_size: Tuple[int, int] = (8, 8)):
        self.grid_size = grid_size
        self.pieces: Dict[Tuple[int, int], Piece] = {}
        self.move_history: List[Dict] = []

    def update_from_detections(self, yolo_results, grid_lines: Tuple[List[int], List[int]]) -> None:
        h_lines, v_lines = grid_lines
        self.clear_pieces()
        for detection in yolo_results[0].boxes:
            cls = int(detection.cls)
            bbox = detection.xyxy[0].cpu().numpy()
            piece_type, color = self._classify_detection(cls)
            position = self._bbox_to_position(bbox, h_lines, v_lines)
            if position:
                self.pieces[position] = Piece(type=piece_type, color=color, position=position)

    def _bbox_to_position(self, bbox: np.ndarray, h_lines: List[int], v_lines: List[int]) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        row = np.searchsorted(h_lines, y_center) - 1
        col = np.searchsorted(v_lines, x_center) - 1
        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            return (row, col)
        return None

    def _classify_detection(self, class_id: int) -> Tuple[str, str]:
        if class_id == 0:
            return 'flecha', 'none'
        elif class_id == 1:
            return 'amazona', 'black'
        elif class_id == 2:
            return 'amazona', 'white'
        else:
            return 'unknown', 'none'

    def clear_pieces(self) -> None:
        self.pieces.clear()

    def to_sgf(self) -> str:
        sgf = "(;GM[Amazons]SZ[8]"
        for (row, col), piece in self.pieces.items():
            pos = self._coord_to_sgf(row, col)
            if piece.type == 'amazona':
                sgf += f"{'AW' if piece.color == 'white' else 'AB'}[{pos}]"
            else:
                sgf += f"AE[{pos}]"
        sgf += ")"
        return sgf

    def _coord_to_sgf(self, row: int, col: int) -> str:
        return f"{chr(97 + col)}{8 - row}"

    def get_amazons_positions(self, color: str) -> List[Tuple[int, int]]:
        return [pos for pos, piece in self.pieces.items()
                if piece.type == 'amazona' and piece.color == color]

    def get_arrows_positions(self) -> List[Tuple[int, int]]:
        return [pos for pos, piece in self.pieces.items()
                if piece.type == 'flecha']

# ==================== Función para detectar la cuadrícula ====================
def detect_grid_hough(warped_image, debug=False):
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    if debug:
        cv2.imshow("Edges", edges)
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
    if len(h_lines) == 9 and len(v_lines) == 9:
        return sorted(h_lines), sorted(v_lines)
    else:
        return np.linspace(0, height, 9), np.linspace(0, width, 9)

# ==================== Funciones para gestionar e imprimir el tablero ====================
def create_board_matrix(board_tracker: BoardTracker) -> List[List[str]]:
    board = [["0" for _ in range(board_tracker.grid_size[1])] 
             for _ in range(board_tracker.grid_size[0])]

    for (row, col), piece in board_tracker.pieces.items():
        if piece.type == 'amazona':
            board[row][col] = "W" if piece.color == 'white' else "B"
        elif piece.type == 'flecha':
            board[row][col] = "A"
    return board

def print_board(board: List[List[str]]) -> None:
    print("Estado del Tablero:")
    for row in board:
        print(" ".join(row))
    print("")

# ==================== Función principal ====================
def main():
    # url = "http://192.168.0.105:8080/video"
    # url = "http://10.209.154.25:8080/video"
    # url = "http://192.168.228.1:8080/video"
    url = "http://192.168.1.4:8080/video"
    output_dir = r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Project\sgf_format"
    os.makedirs(output_dir, exist_ok=True)
    movimientos_path = os.path.join(output_dir, "movimientos.txt")

    # Inicializa last_board antes del loop
    last_board = [["0"] * 8 for _ in range(8)]
    
    cap = cv2.VideoCapture(url)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    cv2.namedWindow("Tablero Amazons", cv2.WINDOW_NORMAL)
    board_tracker = BoardTracker(grid_size=(8,8))

    board_printed_once = False  # Imprime mensaje inicial solo una vez
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
            detected = {}
            for i, marker_id in enumerate(ids):
                detected[marker_id[0]] = np.mean(corners[i][0], axis=0)
            if all(k in detected for k in [0, 1, 2, 3]):
                top_left     = detected[2]
                top_right    = detected[3]
                bottom_right = detected[1]
                bottom_left  = detected[0]
                pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                pts_dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame, M, (800, 800))
                h_lines, v_lines = detect_grid_hough(warped, debug=False)
                with io.StringIO() as buf_out, io.StringIO() as buf_err, \
                        contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    results = model.predict(warped, conf=0.5, verbose=False)
                board_tracker.update_from_detections(results, (h_lines, v_lines))
                current_board = create_board_matrix(board_tracker)

                if current_board != last_board:
                    print_board(current_board)
                    num_arrows = len(board_tracker.get_arrows_positions())
                    num_black = len(board_tracker.get_amazons_positions("black"))
                    num_white = len(board_tracker.get_amazons_positions("white"))
                    print(f"Piezas reconocidas: Flechas: {num_arrows}, Amazonas negras: {num_black}, Amazonas blancas: {num_white}\n")

                    move_str = detect_move(last_board, current_board)
                    if move_str:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        with open(movimientos_path, "a", encoding="utf-8") as f:
                            f.write(f"{timestamp} - {move_str}\n")

                    last_board = current_board

                if not board_printed_once:
                    print("Tablero detectado con éxito (Orientación fija por ID).")
                    board_printed_once = True

                annotated = results[0].plot()
                if h_lines is not None and v_lines is not None:
                    for y in h_lines:
                        cv2.line(annotated, (0, int(y)), (800, int(y)), (0, 255, 0), 2)
                    for x in v_lines:
                        cv2.line(annotated, (int(x), 0), (int(x), 800), (0, 255, 0), 2)
                cv2.imshow("Tablero Amazons", annotated)
                
                
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty("Tablero Amazons", cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
