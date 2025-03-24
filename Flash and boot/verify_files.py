import uos

print("files found in sd card:", uos.listdir("/sd"))
print("files found in flash:", uos.listdir("/flash"))


# Ver capturas de imagenes (experimento cada 5 sec y envio de datos)
print("files found in Snapshots:", uos.listdir("/sd/Snapshots"))
