#
# urequests - micropython-optimized requests module (standalone version)
#
# Nota: No soporta HTTPS ni multipart/form-data automáticamente.
# Para peticiones HTTP sencillas con data= o json=, funciona bien.
#

import usocket

class Response:

    def __init__(self, f):
        self.raw = f
        self.encoding = 'utf-8'
        self._cached = None

    def close(self):
        if self.raw:
            self.raw.close()
            self.raw = None

    @property
    def text(self):
        if self._cached is None:
            self._cached = str(self.content, self.encoding)
        return self._cached

    @property
    def content(self):
        if self._cached is None:
            self._cached = self.raw.read()
        return self._cached

    def json(self):
        import ujson
        return ujson.loads(self.text)

def request(method, url, data=None, json=None, headers={}, stream=None):
    if not url.startswith('http://'):
        raise ValueError("Only 'http' protocol supported")

    # parse url
    proto, dummy, host, path = url.split('/', 3)
    if '/' in host:
        host, path = host.split('/', 1)
    addr = usocket.getaddrinfo(host, 80)[0][-1]
    s = usocket.socket()
    s.connect(addr)
    s.send(b"%b /%b HTTP/1.0\r\n" % (method, path.encode()))
    s.send(b"Host: %b\r\n" % host.encode())

    # If we need to send any headers
    for k in headers:
        s.send(b"%b: %b\r\n" % (k.encode(), headers[k].encode()))

    if json is not None:
        import ujson
        data = ujson.dumps(json)
        s.send(b"Content-Type: application/json\r\n")

    if data:
        s.send(b"Content-Length: %d\r\n" % len(data))
        s.send(b"\r\n")
        s.send(data)
    else:
        s.send(b"\r\n")

    l = s.readline()
    # Example: b'HTTP/1.0 200 OK\r\n'
    l = l.split(None, 2)
    status = int(l[1])

    # Skip response headers
    while True:
        l = s.readline()
        if not l or l == b"\r\n":
            break

    resp = Response(s)
    resp.status_code = status
    resp.reason = None
    return resp

def head(url, **kw):
    return request(b"HEAD", url, **kw)

def get(url, **kw):
    return request(b"GET", url, **kw)

def post(url, **kw):
    return request(b"POST", url, **kw)

def put(url, **kw):
    return request(b"PUT", url, **kw)

def patch(url, **kw):
    return request(b"PATCH", url, **kw)

def delete(url, **kw):
    return request(b"DELETE", url, **kw)
