import ipaddress

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip(request: Request) -> str:
    """Adresse IP réelle du client, utilisée comme clé de limitation de débit.

    Derrière cloudflared, toutes les requêtes arrivent de 127.0.0.1 : sans correction,
    tous les visiteurs partageraient le même quota. Cloudflare met la vraie IP dans
    l'en-tête CF-Connecting-IP. On ne lui fait confiance que si la connexion vient de la
    machine elle-même (donc du tunnel) ; sinon n'importe qui pourrait l'inventer.
    """
    host = get_remote_address(request)
    try:
        from_local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        from_local = False
    if from_local:
        return request.headers.get("cf-connecting-ip", host)
    return host


limiter = Limiter(key_func=client_ip)


class BodySizeLimit:
    """Refuse (413) tout corps de requête plus gros que max_bytes.

    Le max_length du FEN n'est vérifié qu'une fois le JSON entièrement lu en mémoire.
    Ce middleware coupe avant : soit le Content-Length annoncé est trop grand, soit le
    corps envoyé par morceaux (chunked) dépasse en cours de lecture.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited_receive, send)
