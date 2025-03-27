import usocket
import uos
import time
import gc

def post_file_stream(url, file_path, fieldname="image", mimetype="image/jpeg", chunk_size=256):
    # Forzar recolección de basura para liberar RAM
    gc.collect()

    # Parsear la URL (soporta solo http://)
    proto, dummy, host, path = url.split('/', 3)
    if '/' in host:
        host, path = host.split('/', 1)
    addr = usocket.getaddrinfo(host, 80)[0][-1]
    s = usocket.socket()
    s.connect(addr)
    
    # Boundary fijo
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    # Construir cabecera y pie del multipart
    header_str = (
        "--" + boundary + "\r\n" +
        'Content-Disposition: form-data; name="{}"; filename="{}"\r\n'.format(fieldname, file_path.split("/")[-1]) +
        "Content-Type: {}\r\n\r\n".format(mimetype)
    )
    footer_str = "\r\n--" + boundary + "--\r\n"
    
    header = header_str.encode()
    footer = footer_str.encode()
    
    # Obtener tamaño del archivo
    stat = uos.stat(file_path)
    file_size = stat[6]
    total_length = len(header) + file_size + len(footer)
    
    # Enviar línea de petición y cabeceras HTTP
    request_line = "POST /{} HTTP/1.0\r\n".format(path)
    s.send(request_line.encode())
    s.send("Host: {}\r\n".format(host).encode())
    s.send("Content-Type: multipart/form-data; boundary={}\r\n".format(boundary).encode())
    s.send("Content-Length: {}\r\n".format(total_length).encode())
    s.send("\r\n".encode())
    
    # Enviar cabecera multipart
    s.send(header)
    
    # Abrir el archivo y enviar en bloques sin cargarlo completo en RAM
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            s.send(chunk)
    
    # Enviar pie multipart
    s.send(footer)
    
    # Leer la respuesta del servidor en bloques pequeños
    response = b""
    try:
        while True:
            data = s.recv(256)
            if not data:
                break
            response += data
    except Exception as e:
        pass
    s.close()
    return response

def upload_images_streaming():
    SERVER_IP = "192.168.0.100"  # Asegúrate que coincida con la IP de tu PC
    PORT = 5000
    UPLOAD_URL = "http://" + SERVER_IP + ":" + str(PORT) + "/upload"
    
    carpeta = "/sd/Snapshots"
    try:
        archivos = uos.listdir(carpeta)
    except Exception as e:
        print("Error leyendo la carpeta:", e)
        return
    
    for nombre in archivos:
        if nombre.lower().endswith(".jpg"):
            file_path = carpeta + "/" + nombre
            print("Streaming upload:", file_path)
            try:
                resp = post_file_stream(UPLOAD_URL, file_path, chunk_size=256)
                print("Respuesta:", resp)
                time.sleep(0.5)
            except Exception as e:
                print("Error subiendo", nombre, ":", e)

# Ejecuta la función de subida en streaming
upload_images_streaming()
