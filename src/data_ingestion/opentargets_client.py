"""Open Targets Platform GraphQL API client with rate limiting and caching."""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import httpx
from cachetools import TTLCache

from ..config import get_config, get_logger
from .rate_limiter import AdaptiveRateLimiter


logger = get_logger(__name__)


class OpenTargetsAPIError(Exception):
    """Custom exception for Open Targets API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class _RetryableStatusError(Exception):
    """Internal signal that a request failed with a retryable HTTP status (429/5xx)."""

    def __init__(self, status_code: int):
        super().__init__(f"Retryable status code: {status_code}")
        self.status_code = status_code


@dataclass
class OpenTargetsResponse:
    """Structured response from the Open Targets API."""
    success: bool
    data: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0


# GraphQL queries. Field names confirmed via live introspection against
# https://api.platform.opentargets.org/api/v4/graphql (see docs/gene_drug_protein_graph_blueprint.md).
# `pharmacogenomics` takes no pagination args and returns every row for the entity.

SEARCH_QUERY = """
query Search($q: String!, $entityNames: [String!], $size: Int!) {
  search(queryString: $q, entityNames: $entityNames, page: {index: 0, size: $size}) {
    total
    hits {
      id
      entity
      name
      description
      score
    }
  }
}
"""

TARGET_PGX_QUERY = """
query TargetPgx($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    proteinIds { id source }
    pharmacogenomics {
      genotypeId
      phenotypeText
      pgxCategory
      evidenceLevel
      isDirectTarget
      literature
      drugs { drugId drug { id name } }
    }
  }
}
"""

DRUG_PGX_QUERY = """
query DrugPgx($id: String!) {
  drug(chemblId: $id) {
    id
    name
    drugType
    pharmacogenomics {
      genotypeId
      phenotypeText
      pgxCategory
      evidenceLevel
      isDirectTarget
      literature
      target { id approvedSymbol }
    }
  }
}
"""


class OpenTargetsClient:
    """Async client for the public, unauthenticated Open Targets Platform GraphQL API."""

    def __init__(self,
                 base_url: str = "https://api.platform.opentargets.org/api/v4/graphql",
                 timeout: float = 15.0,
                 max_retries: int = 3,
                 requests_per_second: float = 5.0,
                 cache_ttl_seconds: int = 3600,
                 cache_max_size: int = 512):
        """Initialize Open Targets client.

        Args:
            base_url: GraphQL endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            requests_per_second: Client-side rate limit (API has no published limit)
            cache_ttl_seconds: In-memory response cache TTL
            cache_max_size: Maximum cached query/variable combinations
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self.rate_limiter = AdaptiveRateLimiter(
            has_api_key=False,
            default_rate=requests_per_second
        )
        self._cache: TTLCache = TTLCache(maxsize=cache_max_size, ttl=cache_ttl_seconds)

        self._client: Optional[httpx.AsyncClient] = None
        self._session_active = False

        logger.info(f"Initialized OpenTargetsClient for {base_url}")

    @classmethod
    def from_config(cls, config=None) -> "OpenTargetsClient":
        """Create client from configuration.

        Args:
            config: OpenTargetsConfig instance. If None, loads from global app config.

        Returns:
            OpenTargetsClient: Configured client instance
        """
        if config is None:
            app_config = get_config()
            config = app_config.open_targets

        return cls(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            requests_per_second=config.requests_per_second,
            cache_ttl_seconds=config.cache_ttl_seconds,
            cache_max_size=config.cache_max_size,
        )

    async def __aenter__(self) -> "OpenTargetsClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._session_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client is not None:
            await self._client.aclose()
        self._session_active = False

    def _ensure_session(self):
        """Ensure session is active."""
        if not self._session_active or self._client is None:
            raise RuntimeError("OpenTargetsClient not initialized. Use 'async with' context manager.")

    @staticmethod
    def _to_hashable(value: Any) -> Any:
        """Recursively convert lists/dicts to hashable tuples for use as a cache key."""
        if isinstance(value, dict):
            return tuple(sorted((k, OpenTargetsClient._to_hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(OpenTargetsClient._to_hashable(v) for v in value)
        return value

    @classmethod
    def _cache_key(cls, query: str, variables: Dict[str, Any]):
        return (query, cls._to_hashable(variables))

    async def _post_graphql(self, query: str, variables: Dict[str, Any]) -> OpenTargetsResponse:
        """POST a GraphQL query with retry, rate limiting, and caching.

        Args:
            query: GraphQL query string
            variables: GraphQL query variables

        Returns:
            OpenTargetsResponse: Structured response
        """
        self._ensure_session()

        cache_key = self._cache_key(query, variables)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return OpenTargetsResponse(success=True, data=cached, status_code=200, retry_count=0)

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                await self.rate_limiter.acquire()

                response = await self._client.post(
                    self.base_url,
                    json={"query": query, "variables": variables}
                )

                result = self._handle_response(response, attempt)
                if result.success:
                    self._cache[cache_key] = result.data
                return result

            except httpx.RequestError as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                self.rate_limiter.record_error(0)

            except _RetryableStatusError as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} got retryable status {e.status_code}")

            if attempt < self.max_retries:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                await asyncio.sleep(wait_time)

        error_msg = f"Request failed after {self.max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {str(last_exception)}"

        return OpenTargetsResponse(
            success=False, data=None, error_message=error_msg, retry_count=self.max_retries + 1
        )

    def _handle_response(self, response: httpx.Response, attempt: int) -> OpenTargetsResponse:
        """Handle HTTP response, including GraphQL errors returned with HTTP 200.

        Args:
            response: HTTP response
            attempt: Current attempt number

        Returns:
            OpenTargetsResponse: Structured response

        Raises:
            _RetryableStatusError: For 429/5xx responses, to trigger a retry
        """
        status_code = response.status_code

        if status_code == 429:
            self.rate_limiter.record_error(429)
            raise _RetryableStatusError(status_code)

        if status_code >= 500:
            self.rate_limiter.record_error(status_code)
            raise _RetryableStatusError(status_code)

        if 400 <= status_code < 500:
            self.rate_limiter.record_error(status_code)
            return OpenTargetsResponse(
                success=False, data=None,
                error_message=f"Client error {status_code}: {response.text}",
                status_code=status_code, retry_count=attempt
            )

        # 2xx: GraphQL errors can still be present in the body even on HTTP 200
        try:
            body = response.json()
        except ValueError as e:
            self.rate_limiter.record_error(status_code)
            return OpenTargetsResponse(
                success=False, data=None,
                error_message=f"Failed to parse response: {str(e)}",
                status_code=status_code, retry_count=attempt
            )

        if body.get("errors"):
            self.rate_limiter.record_error(status_code)
            messages = "; ".join(err.get("message", "unknown error") for err in body["errors"])
            return OpenTargetsResponse(
                success=False, data=None,
                error_message=f"GraphQL error: {messages}",
                status_code=status_code, retry_count=attempt
            )

        self.rate_limiter.record_success()
        return OpenTargetsResponse(
            success=True, data=body.get("data"), status_code=status_code, retry_count=attempt
        )

    async def search_entities(self, query: str, entity_names: Optional[List[str]] = None,
                               limit: int = 10) -> OpenTargetsResponse:
        """Search for targets/drugs by free text.

        Args:
            query: Search text (e.g. gene symbol or drug name)
            entity_names: Entity types to search, defaults to target + drug
            limit: Maximum number of hits

        Returns:
            OpenTargetsResponse: Search results
        """
        variables = {"q": query, "entityNames": entity_names or ["target", "drug"], "size": limit}
        logger.info(f"Searching Open Targets for: {query}")
        return await self._post_graphql(SEARCH_QUERY, variables)

    async def get_target_with_pharmacogenomics(self, ensembl_id: str) -> OpenTargetsResponse:
        """Fetch a gene/protein target with all known pharmacogenomic drug interactions.

        Args:
            ensembl_id: Ensembl gene ID (e.g. ENSG00000138109)

        Returns:
            OpenTargetsResponse: Target data with pharmacogenomics rows
        """
        logger.info(f"Fetching target pharmacogenomics for: {ensembl_id}")
        return await self._post_graphql(TARGET_PGX_QUERY, {"id": ensembl_id})

    async def get_drug_with_pharmacogenomics(self, chembl_id: str) -> OpenTargetsResponse:
        """Fetch a drug with all known pharmacogenomic gene interactions.

        Args:
            chembl_id: ChEMBL drug ID (e.g. CHEMBL1464)

        Returns:
            OpenTargetsResponse: Drug data with pharmacogenomics rows
        """
        logger.info(f"Fetching drug pharmacogenomics for: {chembl_id}")
        return await self._post_graphql(DRUG_PGX_QUERY, {"id": chembl_id})

    def get_client_status(self) -> Dict[str, Any]:
        """Get comprehensive client status and metrics.

        Returns:
            Dict[str, Any]: Client status information
        """
        return {
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "session_active": self._session_active,
            "cache_size": len(self._cache),
            "rate_limiter": self.rate_limiter.get_metrics(),
        }
