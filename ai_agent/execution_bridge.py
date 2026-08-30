"""
Integrity and model-capability helpers for the LLM/MCP execution bridge.

Two problems this exists to close.

The LLM authors the arguments it passes to the MCP dispatch tool, so it can
alter the payload the risk gates approved. The system prompt asks it not to
mutate strikes, ratios or quantities — but a prompt is a request, not a
constraint, and it is the same shape as the earlier "please send quantities as
strings" instruction that had to be replaced with a cast in code. The gates
approve exactly one structure; an approval hash travels with it and the MCP
tool refuses anything that does not match.

Featherless supports native tool calling only on the Qwen 3 family and
moonshotai/Kimi-K2-Instruct. Llama models fall back to simulated calling via
prompting and never populate `tool_calls`, so an agent pointed at a Llama ID
silently declines every trade. `is_tool_calling_model` makes that a check
rather than a surprise on Monday morning.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


class ApprovalMismatch(RuntimeError):
    """The payload presented for dispatch is not the one the gates approved."""


# Featherless families with native tool calling. Anything else returns no
# tool_calls at all, which reads as "the model declined" rather than as a
# configuration error.
TOOL_CALLING_MODELS = (
    "qwen3",
    "qwen/qwen3",
    "kimi-k2",
    "moonshotai/kimi-k2",
)


def is_tool_calling_model(model_id: str) -> bool:
    """Whether this model can actually return `tool_calls` on Featherless."""
    lowered = (model_id or "").lower()
    return any(marker in lowered for marker in TOOL_CALLING_MODELS)


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """A stable fingerprint of an order payload.

    Serialised with sorted keys so the hash survives the LLM re-serialising the
    JSON, and changes if any strike, side, ratio or quantity is altered.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_approval(payload_json: str, approval_hash: str) -> dict[str, Any]:
    """Return the payload only if it matches the approved fingerprint.

    Raises rather than returning a flag: a mismatch means something between the
    gates and the broker changed the trade, and that must stop the run.
    """
    if not approval_hash:
        raise ApprovalMismatch("No approval hash supplied; refusing to dispatch.")

    try:
        payload = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ApprovalMismatch(f"Payload is not valid JSON: {exc}") from exc

    actual = canonical_payload_hash(payload)
    if actual != approval_hash:
        raise ApprovalMismatch(
            "Payload does not match the gate-approved order. "
            f"expected {approval_hash[:12]}… got {actual[:12]}… — refusing to dispatch."
        )
    return payload
