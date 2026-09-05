"""Fast-mode post-write governance remediation.

In fast mode (the default) enrichment is deferred to core-worker, which PATCHes
the LLM's ``contains_pii`` / ``business_relevance`` onto the already-persisted
row. The ``memclaw.memory.enriched`` consumer then calls
:func:`remediate_after_enrichment` to apply the tenant's configured action on
that free-form signal — the fast-mode counterpart to the synchronous
``GovernanceDecision`` step (strong mode).

The DETERMINISTIC pattern gate already ran synchronously pre-write
(``GovernanceScanContent``), so regex/Luhn/entropy-detectable PII/PCI/secrets
were never persisted in either mode; only the LLM's free-form judgement is
eventually-consistent here (≈ enrichment-deferral latency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core_api.clients.storage_client import get_storage_client
from core_api.services.governance_gate import (
    ACTION_NB_DROP,
    ACTION_NB_KEEP_PRIVATE,
    ACTION_PII_DROP,
    ACTION_PII_FLAG,
    ACTION_PII_MASK,
    emit_governance_audit,
    llm_pii_audit_detail,
    nonbusiness_audit_detail,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemediationOutcome:
    """What remediation did to the row — enough for a caller to follow it.

    ``dropped`` alone was sufficient while the only caller had nothing left to
    do afterwards. A caller that goes on to create rows DERIVED from this one
    needs the rest: ``keep_private`` downgrades the row's visibility, and a
    derived row that keeps the pre-downgrade visibility re-publishes exactly
    the content the policy just made private (#808).
    """

    dropped: bool = False
    visibility: str | None = None
    """The row's new visibility when remediation changed it; ``None`` when it
    was left alone."""


async def _purge_entity_artifacts(sc: Any, memory_id: str, tenant_id: str, action: str) -> None:
    """Remove the graph rows mined out of a memory the policy just dropped.

    H-02. #808 established that a drop must stop the derived rows too, and
    named this case explicitly — "entities mined out of dropped content are the
    same leak in another table". It fixed the INLINE path, by ordering:
    ``_enrich_memory_background`` runs remediation first and its early return
    skips the entity extraction scheduled below it.

    Both non-inline paths schedule extraction independently, at write time, as
    a task that races the verdict — and extraction is one LLM call while the
    verdict needs enrichment plus an event round-trip, so extraction usually
    wins. ``process_entity_extraction`` never re-checked the row, and a
    soft-deleted memory still satisfies the link and relation foreign keys. So
    the names survived, listable tenant-wide through ``/entities`` and
    ``/graph``, with nothing tying them to the drop.

    Not guarded by a marker, unlike the child cascade: any dropped memory may
    have been extracted from, there is no flag on the row saying so, and the
    purge is three targeted deletes keyed on ``memory_id`` — cheap enough to
    run unconditionally rather than gate on something that could drift.

    Failures are LOGGED, not raised, and that is the opposite of what the rest
    of this module does — so it needs its reason stated.

    An earlier draft let this propagate "matching every other unapplied-policy
    path in this module". That was wrong about the caller. The other paths run
    under ``_enrich_memory_background``, where a raise becomes a
    ``BackgroundTaskLog`` row. This one also runs under
    ``consumer.handle_memory_enriched``, which has no guard around it, and the
    Pub/Sub dispatcher NACKS on a handler exception — a documented, load-bearing
    invariant. So a raise here redelivers the same event, re-runs the whole drop
    branch, and emits a SECOND ``critical=True`` audit for a memory that was
    already dropped. Repeatedly. Duplicate destructive entries in a
    tamper-evident log, for a row nothing further can be done to.

    The trade the other way is bounded: the memory itself is already gone, so
    the content is not live. What remains is graph rows, and an ERROR naming the
    memory is enough to purge them by hand. A transient failure does not even
    reach here — ``purge_entity_artifacts`` is marked idempotent, so the client
    retries 5xx and timeouts on its own.
    """
    try:
        counts = await sc.purge_entity_artifacts(tenant_id, memory_id)
    except Exception:
        logger.exception(
            "governance: %s dropped memory %s but its entity/relation rows were NOT "
            "removed; the names mined from that content are still listable and need "
            "purging by hand",
            action,
            memory_id,
        )
        return
    if any(counts.get(k) for k in ("links", "relations", "entities")):
        logger.info(
            "governance: %s purged graph rows for %s (links=%s relations=%s entities=%s)",
            action,
            memory_id,
            counts.get("links"),
            counts.get("relations"),
            counts.get("entities"),
        )


async def remediate_after_enrichment(memory: dict, cfg: Any) -> RemediationOutcome:
    """Apply LLM-signal governance to a fast-mode row after enrichment landed.

    Returns what was done. A caller that only needs "should I stop?" reads
    ``.dropped``; one that creates derived rows must also honour
    ``.visibility``. No-op when governance is disabled or the signals are clean.
    """
    pii_cfg = cfg.governance_pii
    nb_cfg = cfg.governance_non_business
    if not pii_cfg.enabled and not nb_cfg.enabled:
        return RemediationOutcome()

    md = memory.get("metadata_") or memory.get("metadata") or {}
    content = memory.get("content") or ""
    tenant_id = memory.get("tenant_id")
    agent_id = memory.get("agent_id")
    raw_id = memory.get("id")
    if raw_id is None:
        # A malformed enriched-event payload without an id would otherwise
        # soft-delete "None" and stamp resource_id="None" on every audit row.
        logger.warning("governance: remediate_after_enrichment called with memory missing 'id'; skipping")
        return RemediationOutcome()
    memory_id = str(raw_id)
    sc = get_storage_client()

    # ── PII (LLM free-form signal) ──
    if pii_cfg.enabled and md.get("contains_pii"):
        pii_types = md.get("pii_types") or []
        if pii_cfg.action == "drop":
            # Audit BEFORE the destructive delete (mirrors GovernanceScanContent's
            # audit-before-mutate): a delete that succeeds before a failing audit
            # would leave an untracked deletion in the tamper-evident log, whereas
            # an audit-then-failed-delete leaves a remediable "intended to drop" trace.
            await emit_governance_audit(
                tenant_id=tenant_id,
                agent_id=agent_id,
                action=ACTION_PII_DROP,
                detail=llm_pii_audit_detail(ACTION_PII_DROP, pii_types, content, "fast"),
                resource_id=memory_id,
                # Destructive: the soft-delete below removes the row, so this
                # audit is the only trace — must survive queue overflow.
                critical=True,
            )
            await sc.soft_delete_memory(memory_id, tenant_id)
            logger.info("governance: dropped fast-mode memory %s (pii)", memory_id)
            await _purge_entity_artifacts(sc, memory_id, tenant_id, ACTION_PII_DROP)
            return RemediationOutcome(dropped=True)
        # mask/flag: the LLM gives no offsets to redact a free-form span, and in
        # fast mode the row is already persisted — so a "mask"-configured tenant
        # can only be flagged here. Keep the action truthful (flag), but record
        # the configured intent in the detail so compliance can tell this apart
        # from a genuine flag policy.
        await emit_governance_audit(
            tenant_id=tenant_id,
            agent_id=agent_id,
            action=ACTION_PII_FLAG,
            detail=llm_pii_audit_detail(
                ACTION_PII_FLAG,
                pii_types,
                content,
                "fast",
                configured_action=ACTION_PII_MASK if pii_cfg.action == "mask" else None,
            ),
            resource_id=memory_id,
        )

    # ── Business-vs-personal disposition ──
    if nb_cfg.enabled and md.get("business_relevance") == "personal":
        if nb_cfg.disposition == "drop":
            # Audit before the destructive delete (see the PII-drop branch above).
            await emit_governance_audit(
                tenant_id=tenant_id,
                agent_id=agent_id,
                action=ACTION_NB_DROP,
                detail=nonbusiness_audit_detail(ACTION_NB_DROP, content, "fast"),
                resource_id=memory_id,
                # Destructive: see the PII-drop branch — audit is the only trace.
                critical=True,
            )
            await sc.soft_delete_memory(memory_id, tenant_id)
            logger.info("governance: dropped fast-mode memory %s (non-business)", memory_id)
            await _purge_entity_artifacts(sc, memory_id, tenant_id, ACTION_NB_DROP)
            return RemediationOutcome(dropped=True)
        if nb_cfg.disposition == "keep_private":
            await sc.update_memory(memory_id, tenant_id, {"visibility": "scope_agent"})
            await emit_governance_audit(
                tenant_id=tenant_id,
                agent_id=agent_id,
                action=ACTION_NB_KEEP_PRIVATE,
                detail=nonbusiness_audit_detail(ACTION_NB_KEEP_PRIVATE, content, "fast"),
                resource_id=memory_id,
            )
            # Reported so a caller creating derived rows can carry the
            # downgrade to them instead of publishing the same content wider.
            return RemediationOutcome(visibility="scope_agent")
    return RemediationOutcome()
