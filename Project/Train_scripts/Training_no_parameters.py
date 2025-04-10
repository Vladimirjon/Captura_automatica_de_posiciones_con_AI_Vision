import torch
from ultralytics import YOLO

original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
model = YOLO("yolov8n.pt")

# Especficar el modelo a entrenar, la direccion de data.yml, epocas
# Se puede especificar el batch size, el learning rate y el optimizer
model.train(
    data=r"C:\Users\johan\Desktop\Universidad\Proyectos Intersemestrales\Captura_automatica_de_posiciones_con_AI_Vision\Models\Amazons game recognition.v3-v3_amazons_recongnition.yolov8\data.yaml",
    epochs=100
)
