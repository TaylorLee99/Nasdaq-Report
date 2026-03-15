"""HTTP client abstraction and SEC submissions/archive fetchers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RetryableHttpError(RuntimeError):
    """Raised when an HTTP request can be retried."""


class HttpJsonClient(Protocol):
    """Abstract JSON HTTP client."""

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        """Fetch a JSON object from the provided URL."""


@dataclass(frozen=True)
class HttpBinaryResponse:
    """Binary HTTP response payload and headers."""

    content: bytes
    content_type: str | None
    headers: Mapping[str, str]


class HttpBinaryClient(Protocol):
    """Abstract binary HTTP client."""

    def get_binary(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> HttpBinaryResponse:
        """Fetch a binary response from the provided URL."""


class UrllibJsonHttpClient:
    """Default HTTP client using the Python standard library."""

    def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, object]:
        """Fetch and decode a JSON object."""

        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504}:
                raise RetryableHttpError(str(exc)) from exc
            raise
        except URLError as exc:
            raise RetryableHttpError(str(exc)) from exc
        return json.loads(payload)


class UrllibBinaryHttpClient:
    """Binary HTTP client using the Python standard library."""

    def get_binary(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> HttpBinaryResponse:
        """Fetch and return raw response bytes."""

        request = Request(url, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type")
                header_map = {key: value for key, value in response.headers.items()}
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504}:
                raise RetryableHttpError(str(exc)) from exc
            raise
        except URLError as exc:
            raise RetryableHttpError(str(exc)) from exc
        return HttpBinaryResponse(
            content=content,
            content_type=content_type,
            headers=header_map,
        )


class RateLimiter:
    """Simple tokenless rate limiter based on minimum request interval."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        monotonic_func: Callable[[], float] = monotonic,
        sleep_func: Callable[[float], None] = sleep,
    ) -> None:
        self._minimum_interval = 1.0 / requests_per_second
        self._monotonic = monotonic_func
        self._sleep = sleep_func
        self._last_request_at = 0.0

    def wait(self) -> None:
        """Sleep if the previous request was too recent."""

        now = self._monotonic()
        elapsed = now - self._last_request_at
        remaining = self._minimum_interval - elapsed
        if remaining > 0:
            self._sleep(remaining)
        self._last_request_at = self._monotonic()


class SecSubmissionsClient:
    """Rate-limited, retrying client for the SEC submissions endpoint."""

    def __init__(
        self,
        http_client: HttpJsonClient,
        *,
        base_url: str,
        user_agent: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        rate_limiter: RateLimiter,
        sleep_func: Callable[[float], None] = sleep,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._rate_limiter = rate_limiter
        self._sleep = sleep_func

    def fetch_submissions(self, cik: str) -> dict[str, object]:
        """Fetch the submissions payload for a CIK."""

        url = f"{self._base_url}/CIK{cik.zfill(10)}.json"
        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        attempt = 0
        while True:
            self._rate_limiter.wait()
            try:
                return self._http_client.get_json(
                    url,
                    headers=headers,
                    timeout_seconds=self._timeout_seconds,
                )
            except RetryableHttpError:
                if attempt >= self._max_retries:
                    raise
                attempt += 1
                self._sleep(self._backoff_seconds * attempt)


class SecArchiveClient:
    """Rate-limited, retrying client for SEC archive documents."""

    def __init__(
        self,
        http_client: HttpBinaryClient,
        *,
        user_agent: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        rate_limiter: RateLimiter,
        sleep_func: Callable[[float], None] = sleep,
    ) -> None:
        self._http_client = http_client
        self._user_agent = user_agent
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._rate_limiter = rate_limiter
        self._sleep = sleep_func

    def download(self, url: str) -> HttpBinaryResponse:
        """Download an SEC archive document with retries."""

        headers = {"User-Agent": self._user_agent, "Accept": "*/*"}
        attempt = 0
        while True:
            self._rate_limiter.wait()
            try:
                return self._http_client.get_binary(
                    url,
                    headers=headers,
                    timeout_seconds=self._timeout_seconds,
                )
            except RetryableHttpError:
                if attempt >= self._max_retries:
                    raise
                attempt += 1
                self._sleep(self._backoff_seconds * attempt)
