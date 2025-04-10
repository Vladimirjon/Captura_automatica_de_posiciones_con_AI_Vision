import torch

# Parcheo de weights para evitar error de "weights_only"
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

if __name__ == '__main__':
    original_torch_load = torch.load
    torch.load = patched_torch_load

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    # Parametros de entrenamiento
    model.train(
        # Especificar la ruta del conjunto de datos
        data=r"C:\Users\johan\Desktop\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Models\Amazons game recognition.v4i.yolov8\data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0  # 0 indica tu GPU NVIDIA
    )
