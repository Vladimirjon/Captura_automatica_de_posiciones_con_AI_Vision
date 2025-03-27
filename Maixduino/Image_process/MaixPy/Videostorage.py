import sensor, image, lcd, time, uos

lcd.init()

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

iteration = 0
while True:
    iteration += 1
    print("Iteración:", iteration)
    img = sensor.snapshot()                         # Captura la imagen
    timestamp = time.ticks_ms()                       # Obtiene el tiempo actual
    filename = "/sd/Snapshots/img_{}.jpg".format(timestamp)  # Ruta en la carpeta Snapshots
    img.save(filename)                                # Guarda la imagen en la SD
    print("Guardado correctamente:", filename)
    lcd.display(img)                                  # Muestra la imagen en la LCD
    time.sleep(0.1)                                   # Espera 5 segundos
