"""Kalshi request signing (Ed25519 or RSA-PSS), per docs.kalshi.com API Keys."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kalshi_quant.shared.config import load_dotenv
from kalshi_quant.shared.utilities.timeutil import now_ms


@dataclass
class Credentials:
    key_id: str
    private_key: object

    @classmethod
    def from_environment(cls) -> "Credentials | None":
        """Read KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH. Returns None when unset."""
        load_dotenv()
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            return None
        pem = Path(key_path).expanduser().read_bytes()
        return cls(key_id, serialization.load_pem_private_key(pem, password=None))

    def sign(self, text: str) -> str:
        message = text.encode()
        if isinstance(self.private_key, Ed25519PrivateKey):
            signature = self.private_key.sign(message)
        else:
            signature = self.private_key.sign(
                message,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return base64.b64encode(signature).decode()

    def headers(self, method: str, url: str) -> dict[str, str]:
        """Sign timestamp + method + path. The query string is excluded from the signed path."""
        timestamp = str(now_ms())
        path = urlsplit(url).path
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": self.sign(timestamp + method + path),
        }
