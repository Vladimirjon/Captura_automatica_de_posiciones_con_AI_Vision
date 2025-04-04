import torch

# Define la función parcheada de torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

if __name__ == '__main__':
    # Esto debe estar dentro del bloque main para evitar problemas de multiprocessing en Windows
    original_torch_load = torch.load
    torch.load = patched_torch_load

    from ultralytics import YOLO

    # Cargar el modelo
    model = YOLO("yolov8n.pt")

    # Especificar el entrenamiento
    model.train(
        data=r"C:\Users\johan\OneDrive\Escritorio\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Models\Amazons game recognition.v2i.yolov8\data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0  # 0 indica tu GPU NVIDIA
    )
