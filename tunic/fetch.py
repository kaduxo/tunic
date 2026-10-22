"""Bounded public HTTPS GET. No cookies, no private or loopback targets."""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

_MAX_BYTES = 200_000
_TIMEOUT = 15.0
_BLOCKED_NAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)


class FetchError(ValueError):
    """User-facing fetch failure. The message has no secrets."""


def public_https_url(raw: str) -> str:
    """Return a cleaned https URL, or raise FetchError."""
    if not isinstance(raw, str) or not raw.strip():
        raise FetchError("fetch requires a url string")
    text = raw.strip()
    if any(ch.isspace() for ch in text) or len(text) > 2048:
        raise FetchError("url is not usable")
    parsed = urlparse(text)
    if parsed.scheme != "https":
        raise FetchError("only https URLs are allowed")
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        raise FetchError("url must not include credentials")
    host = parsed.hostname
    if not host:
        raise FetchError("url has no host")
    if _host_blocked(host):
        raise FetchError("url host is not a public address")
    return parsed.geturl()


def get_public(url: str) -> str:
    """GET a public https page and return text. HTML is stripped."""
    cleaned = public_https_url(url)
    opener = urllib.request.build_opener(_RedirectGuard)
    request = urllib.request.Request(
        cleaned,
        headers={
            "User-Agent": "tunic/0.1",
            "Accept": "text/*, application/json, application/xhtml+xml",
        },
        method="GET",
    )
    host = urlparse(cleaned).netloc or cleaned
    try:
        with opener.open(request, timeout=_TIMEOUT) as response:
            final = response.geturl() if hasattr(response, "geturl") else cleaned
            public_https_url(final)
            content_type = ""
            if hasattr(response, "headers") and response.headers:
                content_type = str(response.headers.get("Content-Type") or "")
            data = _read_capped(response)
    except FetchError:
        raise
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} from {host}") from None
    except urllib.error.URLError as exc:
        raise FetchError(f"could not reach {host}: {exc.reason}") from None
    except TimeoutError as exc:
        raise FetchError(f"timed out talking to {host}") from exc
    except OSError as exc:
        raise FetchError(f"could not fetch {host}: {exc}") from None
    return _decode_body(data, content_type)


def _host_blocked(host: str) -> bool:
    name = host.strip("[]").lower().rstrip(".")
    if name in _BLOCKED_NAMES or name.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        ip = None
    if ip is not None:
        return _ip_blocked(ip)
    try:
        infos = socket.getaddrinfo(name, 443, type=socket.SOCK_STREAM)
    except OSError:
        return True
    if not infos:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            found = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if _ip_blocked(found):
            return True
    return False


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return not ip.is_global


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_capped(response) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < _MAX_BYTES:
        piece = response.read(min(8192, _MAX_BYTES - total + 1))
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
        if total > _MAX_BYTES:
            break
    data = b"".join(chunks)
    return data[:_MAX_BYTES]


def _decode_body(data: bytes, content_type: str) -> str:
    if b"\x00" in data[:8192]:
        raise FetchError("response looks binary")
    text = data.decode("utf-8", errors="replace")
    lowered = (content_type or "").lower()
    if "html" in lowered or (not lowered and text.lstrip()[:15].lower().startswith(("<!doctype", "<html"))):
        return _html_to_text(text)
    return text


class _HTMLText(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        stripped = " ".join(data.split())
        if stripped:
            self.parts.append(stripped + " ")


def _html_to_text(html: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except (ValueError, TypeError, AssertionError):
        return " ".join(html.split())
    lines = []
    for raw in "".join(parser.parts).splitlines():
        line = " ".join(raw.split())
        if line:
            lines.append(line)
    return "\n".join(lines)
