import gc, sys, os, lcd, image
from Maix import GPIO
from fpioa_manager import fm

try:
    # --- CONFIGURACIÓN DE PINES / TEST MODE (si es necesario) ---
    test_pin = 16
    fm.fpioa.set_function(test_pin, fm.fpioa.GPIO7)
    test_gpio = GPIO(GPIO.GPIO7, GPIO.IN, GPIO.PULL_UP)

    # Ejemplo: pin 17 para controlar la pantalla
    fm.fpioa.set_function(17, fm.fpioa.GPIO0)
    lcd_en = GPIO(GPIO.GPIO0, GPIO.OUT)
    lcd_en.value(0)

    # --- INICIALIZAR LCD ---
    lcd.init()
    gc.collect()

    # --- OPCIONAL: MODO TEST (si test_gpio = 0) ---
    if test_gpio.value() == 0:
        print("PIN 16 pulled down, entrar en 'Test Mode'")
        lcd.clear(lcd.PINK)
        lcd.draw_string(lcd.width()//2 - 68, lcd.height()//2 - 4, "Test Mode", lcd.WHITE, lcd.PINK)
        import sensor
        sensor.reset()
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.QVGA)
        sensor.run(1)
        lcd.freq(16000000)
        while True:
            img = sensor.snapshot()
            lcd.display(img)
    else:
        # --- TU PANTALLA ROJA Y TEXTO PERSONALIZADO ---
        loading = image.Image(size=(lcd.width(), lcd.height()))
        loading.draw_rectangle((0, 0, lcd.width(), lcd.height()), fill=True, color=(255, 0, 0))

        top_text_y = lcd.height() // 6
        epn_y = lcd.height() // 2 - 20
        bottom_text_y = lcd.height() * 2 // 3

        info = "AI Vision en Amazon"
        loading.draw_string(
            int(lcd.width()//2 - len(info)*5),
            top_text_y, 
            info, 
            color=(255,255,255), 
            scale=2
        )

        # Dibujo E-P-N
        letter_height = 40
        letter_width = 25
        gap = 10
        total_width = 3 * letter_width + 2 * gap
        start_x = (lcd.width() - total_width) // 2
        start_y = epn_y
        color_epn = (255, 255, 255)
        thickness = 2
        # E
        loading.draw_line(start_x, start_y, start_x, start_y + letter_height, color=color_epn, thickness=thickness)
        loading.draw_line(start_x, start_y, start_x + letter_width, start_y, color=color_epn, thickness=thickness)
        loading.draw_line(start_x, start_y + letter_height//2, start_x + letter_width, start_y + letter_height//2, color=color_epn, thickness=thickness)
        loading.draw_line(start_x, start_y + letter_height, start_x + letter_width, start_y + letter_height, color=color_epn, thickness=thickness)
        # P
        xp = start_x + letter_width + gap
        loading.draw_line(xp, start_y, xp, start_y + letter_height, color=color_epn, thickness=thickness)
        loading.draw_line(xp, start_y, xp + letter_width, start_y, color=color_epn, thickness=thickness)
        loading.draw_line(xp, start_y + letter_height//2, xp + letter_width, start_y + letter_height//2, color=color_epn, thickness=thickness)
        loading.draw_line(xp + letter_width, start_y, xp + letter_width, start_y + letter_height//2, color=color_epn, thickness=thickness)
        # N
        xn = xp + letter_width + gap
        loading.draw_line(xn, start_y, xn, start_y + letter_height, color=color_epn, thickness=thickness)
        loading.draw_line(xn + letter_width, start_y, xn + letter_width, start_y + letter_height, color=color_epn, thickness=thickness)
        loading.draw_line(xn, start_y, xn + letter_width, start_y + letter_height, color=color_epn, thickness=thickness)

        project_name = "Captura de Posiciones automatica"
        user_name = "Johann Pasquel"
        loading.draw_string(
            int(lcd.width()//2 - len(project_name)*3),
            bottom_text_y,
            project_name,
            color=(255,255,255),
            scale=1
        )
        loading.draw_string(
            int(lcd.width()//2 - len(user_name)*3),
            bottom_text_y + 20,
            user_name,
            color=(255,255,255),
            scale=1
        )

        lcd.display(loading)
        del loading, info, project_name, user_name
        gc.collect()

    # Aquí puedes continuar con tu programa principal...

except Exception as e:
    # --- Fallback: si algo falla, mostrar "Welcome to MaixPy" ---
    import lcd
    lcd.init()
    lcd.clear(color=(255,0,0))
    lcd.draw_string(lcd.width()//2 - 68, lcd.height()//2 - 4, "Welcome to MaixPy", lcd.WHITE, lcd.RED)
    print("Error en la inicialización principal:", e)

finally:
    gc.collect()
