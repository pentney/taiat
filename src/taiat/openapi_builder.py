"""
Build an AgentGraphNodeSet directly from an OpenAPI specification.

Each selected endpoint becomes an AgentGraphNode whose callable makes the
corresponding HTTP request. Inputs are derived from path/query parameters
and request body; outputs are derived from the 2xx response schema.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet, State
from taiat.graph_architect import AgentRegistry


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def _load_spec(source: str | Path | dict) -> dict:
    """Return a parsed OpenAPI spec dict from a file path, URL, or existing dict."""
    if isinstance(source, dict):
        return source
    source = str(source)
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            raw = resp.read().decode()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return _load_yaml(raw)
    else:
        path = Path(source)
        raw = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            return _load_yaml(raw)
        return json.loads(raw)


def _load_yaml(text: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML specs: pip install pyyaml"
        ) from exc
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Schema / $ref helpers
# ---------------------------------------------------------------------------

def _resolve_ref(spec: dict, ref: str) -> dict:
    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for part in parts:
        node = node[part]
    return node


def _schema_to_agent_data(
    spec: dict,
    name: str,
    schema: dict,
    *,
    description: str = "",
    location: str = "",
) -> AgentData:
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    params: dict[str, Any] = {}
    if location:
        params["in"] = location
    if "type" in schema:
        params["type"] = schema["type"]
    if "properties" in schema:
        params["properties"] = list(schema["properties"].keys())
    return AgentData(
        name=name,
        description=description or schema.get("description", ""),
        parameters=params,
    )


# ---------------------------------------------------------------------------
# HTTP callable factory
# ---------------------------------------------------------------------------

def _make_http_callable(
    base_url: str,
    path: str,
    method: str,
    path_param_names: list[str],
    query_param_names: list[str],
    has_body: bool,
    output_name: str,
) -> Callable[[State], State]:
    """Return a State → State callable that performs the HTTP request."""

    def call(state: State) -> State:
        data = state["data"]

        url_path = path
        for p in path_param_names:
            value = data[p].data if p in data else ""
            url_path = url_path.replace(f"{{{p}}}", urllib.parse.quote(str(value), safe=""))

        query: dict[str, str] = {
            q: str(data[q].data)
            for q in query_param_names
            if q in data and data[q].data is not None
        }

        full_url = base_url.rstrip("/") + url_path
        if query:
            full_url += "?" + urllib.parse.urlencode(query)

        body: bytes | None = None
        headers: dict[str, str] = {}
        if has_body and "request_body" in data and data["request_body"].data is not None:
            body = json.dumps(data["request_body"].data).encode()
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            full_url, data=body, headers=headers, method=method.upper()
        )
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode()
            try:
                parsed: Any = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
        except urllib.error.HTTPError as exc:
            parsed = {"error": str(exc), "code": exc.code}

        state["data"][output_name] = AgentData(
            name=output_name,
            description=f"Response from {method.upper()} {path}",
            data=parsed,
        )
        return state

    call.__name__ = output_name
    return call


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class OpenAPIGraphBuilder:
    """
    Build an AgentGraphNodeSet and AgentRegistry from an OpenAPI specification.

    Usage::

        builder = OpenAPIGraphBuilder("petstore.yaml")
        node_set, registry = builder.build()

    Each endpoint in the spec becomes an AgentGraphNode. The node's callable
    makes the HTTP request and stores the response in ``state["data"]``.

    Args:
        spec: OpenAPI spec as a file path (str/Path), URL, or already-parsed dict.
        base_url: Override the base URL from the spec's ``servers`` block.
    """

    def __init__(self, spec: str | Path | dict, base_url: Optional[str] = None):
        self._spec = _load_spec(spec)
        if base_url is not None:
            self._base_url = base_url.rstrip("/")
        else:
            servers = self._spec.get("servers", [])
            self._base_url = servers[0]["url"].rstrip("/") if servers else ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        include_paths: Optional[list[str]] = None,
        include_methods: Optional[list[str]] = None,
        include_tags: Optional[list[str]] = None,
    ) -> tuple[AgentGraphNodeSet, AgentRegistry]:
        """
        Parse the spec and return a fully resolved graph and registry.

        Args:
            include_paths: Restrict to these URL paths (e.g. ``["/pets", "/pets/{id}"]``).
                           Defaults to all paths.
            include_methods: Restrict to these HTTP methods (case-insensitive).
                             Defaults to GET, POST, PUT, PATCH, DELETE.
            include_tags: Restrict to endpoints tagged with at least one of these tags.
                          Defaults to all tags.

        Returns:
            ``(AgentGraphNodeSet, AgentRegistry)`` — the node set has callables
            already attached; the registry holds the same callables keyed by
            operation ID so they can be inspected or augmented.
        """
        allowed_methods = {
            m.lower()
            for m in (include_methods or ["get", "post", "put", "patch", "delete"])
        }

        registry = AgentRegistry()
        nodes: list[AgentGraphNode] = []

        for path, path_item in self._spec.get("paths", {}).items():
            if include_paths and path not in include_paths:
                continue
            if not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                if method not in allowed_methods:
                    continue
                if not isinstance(operation, dict):
                    continue
                if include_tags:
                    op_tags = operation.get("tags", [])
                    if not any(t in include_tags for t in op_tags):
                        continue

                node, fn = self._build_node(path, method, operation)
                registry.register(node.name, fn)
                nodes.append(node)

        return AgentGraphNodeSet(nodes=nodes), registry

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_node(
        self, path: str, method: str, operation: dict
    ) -> tuple[AgentGraphNode, Callable]:
        operation_id = (
            operation.get("operationId")
            or f"{method}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
        )
        description = operation.get("summary") or operation.get("description") or ""

        inputs, path_params, query_params = self._parse_parameters(operation)
        has_body, body_input = self._parse_request_body(operation)
        if body_input:
            inputs.append(body_input)

        output_name = f"{operation_id}_response"
        outputs = self._parse_responses(operation, path, method, output_name)

        fn = _make_http_callable(
            self._base_url, path, method, path_params, query_params, has_body, output_name
        )

        node = AgentGraphNode(
            name=operation_id,
            description=description,
            function=fn,
            inputs=inputs,
            outputs=outputs,
        )
        return node, fn

    def _parse_parameters(
        self, operation: dict
    ) -> tuple[list[AgentData], list[str], list[str]]:
        inputs: list[AgentData] = []
        path_params: list[str] = []
        query_params: list[str] = []

        for param in operation.get("parameters", []):
            if "$ref" in param:
                param = _resolve_ref(self._spec, param["$ref"])
            loc = param.get("in", "query")
            name = param["name"]
            schema = param.get("schema", {})
            ad = _schema_to_agent_data(
                self._spec,
                name,
                schema,
                description=param.get("description", ""),
                location=loc,
            )
            inputs.append(ad)
            if loc == "path":
                path_params.append(name)
            elif loc == "query":
                query_params.append(name)

        return inputs, path_params, query_params

    def _parse_request_body(
        self, operation: dict
    ) -> tuple[bool, Optional[AgentData]]:
        request_body = operation.get("requestBody")
        if not request_body:
            return False, None

        rb_desc = request_body.get("description", "Request body")
        content = request_body.get("content", {})
        schema: dict = {}
        for _ct, ct_info in content.items():
            if "schema" in ct_info:
                schema = ct_info["schema"]
                break

        ad = _schema_to_agent_data(
            self._spec, "request_body", schema, description=rb_desc, location="body"
        )
        return True, ad

    def _parse_responses(
        self, operation: dict, path: str, method: str, output_name: str
    ) -> list[AgentData]:
        for status, resp_obj in operation.get("responses", {}).items():
            if not str(status).startswith("2"):
                continue
            if "$ref" in resp_obj:
                resp_obj = _resolve_ref(self._spec, resp_obj["$ref"])
            resp_desc = resp_obj.get("description", "")
            content = resp_obj.get("content", {})
            schema: dict = {}
            for _ct, ct_info in content.items():
                if "schema" in ct_info:
                    schema = ct_info["schema"]
                    break
            return [
                _schema_to_agent_data(
                    self._spec, output_name, schema, description=resp_desc
                )
            ]

        return [
            AgentData(
                name=output_name,
                description=f"Response from {method.upper()} {path}",
            )
        ]
