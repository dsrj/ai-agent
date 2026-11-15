"""Read-only GitHub access through the "CloudOps Hub - Read Sources" GitHub App.

The app's private key lives in Key Vault. Each use mints a short-lived (1 hour)
installation token. There is no long-lived personal access token fallback.
"""

import base64
import json
import time

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

API = "https://api.github.com"
PRIVATE_KEY_SECRET = "github-read-app-private-key"
# Any repository the installation covers identifies it when no installation ID is set.
DISCOVERY_REPOSITORY = "almbrand/landingzone-service-catalog"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def app_jwt(app_id: int, private_key: str, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time()) - 60
    payload = {"iat": issued, "exp": issued + 540, "iss": str(app_id)}
    signing_input = ".".join(
        _b64(json.dumps(part, separators=(",", ":")).encode())
        for part in ({"alg": "RS256", "typ": "JWT"}, payload)
    )
    key = serialization.load_pem_private_key(private_key.encode(), password=None)
    signature = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64(signature)}"


def installation_token(
    app_id: int,
    private_key: str,
    installation_id: int = 0,
    *,
    transport: httpx.BaseTransport | None = None,
) -> str:
    headers = {**HEADERS, "Authorization": f"Bearer {app_jwt(app_id, private_key)}"}
    with httpx.Client(
        base_url=API,
        headers=headers,
        timeout=30,
        follow_redirects=False,
        transport=transport,
    ) as client:
        if not installation_id:
            response = client.get(f"/repos/{DISCOVERY_REPOSITORY}/installation")
            if response.status_code != 200:
                raise RuntimeError(
                    f"Read Sources GitHub App is not installed on {DISCOVERY_REPOSITORY} "
                    f"(HTTP {response.status_code})"
                )
            installation_id = int(response.json()["id"])
        response = client.post(f"/app/installations/{int(installation_id)}/access_tokens")
        if response.status_code != 201:
            raise RuntimeError(
                f"Read Sources GitHub App token request failed (HTTP {response.status_code}); "
                "check the app ID, installation ID and private key"
            )
        token = response.json().get("token")
    if not token:
        raise RuntimeError("Read Sources GitHub App returned no token")
    return token


def read_token(get_secret, app_id: int, installation_id: int = 0) -> str:
    """Return a short-lived Read Sources app installation token.

    ``get_secret(name)`` returns the Key Vault secret value (or raises).
    """
    if not app_id:
        raise RuntimeError(
            "GitHub reads require the Read Sources GitHub App; set GITHUB_READ_APP_ID"
        )
    private_key = get_secret(PRIVATE_KEY_SECRET)
    if not private_key:
        raise RuntimeError(f"Key Vault secret {PRIVATE_KEY_SECRET} is empty")
    return installation_token(app_id, private_key, installation_id)
