import sensor, image, lcd, time, uos, machine

# Inicializa la pantalla LCD
lcd.init()

# Inicializa la cámara
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Formato RGB565
sensor.set_framesize(sensor.QVGA)      # Resolución 320x240
sensor.skip_frames(time=2000)          # Espera 2 segundos para estabilizar la cámara

# Inicializa la tarjeta SD
# Dependiendo de tu placa MaixDuino, el método de inicialización puede variar.
# En muchos casos se puede usar machine.SD() si el hardware ya está configurado.
try:
    sd = machine.SD()            # Crea el objeto SD (asegúrate de que tu placa lo soporte)
    uos.mount(sd, "/sd")         # Monta la SD en la ruta /sd
    print("SD montada correctamente")
except Exception as e:
    print("Error al montar SD:", e)

# Bucle principal: captura, guarda y muestra la imagen cada 5 segundos
while True:
    img = sensor.snapshot()                          # Captura una imagen
    timestamp = time.ticks_ms()                        # Obtiene el tiempo actual en milisegundos
    filename = "/sd/img_{}.jpg".format(timestamp)       # Crea un nombre único para el archivo
    img.save(filename)                                 # Guarda la imagen en la SD
    print("Guardado correctamente:", filename)
    lcd.display(img)                                   # Muestra la imagen en la pantalla LCD
    time.sleep(5000)                                   # Espera 5 segundos antes de repetir
