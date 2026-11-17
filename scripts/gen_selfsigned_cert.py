"""
Generates self-signed SSL certificate for local HTTPS serving.
Enables Android Chrome to open /camera over local Wi-Fi / hotspot.
"""

import os
import ssl
from pathlib import Path

def generate_self_signed_cert(cert_dir: Path):
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists():
        print(f"Existing certificates found at {cert_path}")
        return cert_path, key_path

    print("Generating self-signed certificate...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"ForestGuard-AI-Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"ForestGuard")
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"Generated certificates:\n  Cert: {cert_path}\n  Key: {key_path}")
        return cert_path, key_path

    except ImportError:
        print("Notice: 'cryptography' package not installed in environment yet.")
        return None, None

if __name__ == "__main__":
    run_dir = Path(__file__).resolve().parent.parent / "run"
    generate_self_signed_cert(run_dir)
