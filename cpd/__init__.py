import os

# Some networks re-sign TLS; point HF/requests at the system bundle when it exists.
_ca = "/etc/ssl/certs/ca-certificates.crt"
if os.path.exists(_ca):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
    os.environ.setdefault("SSL_CERT_FILE", _ca)
