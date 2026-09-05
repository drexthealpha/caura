"""Tests for fast-mode post-write governance remediation (eToro).

The fast-mode counterpart to ``GovernanceDecision``: in the default fast mode
enrichment is deferred to the worker, which PATCHes the LLM's ``contains_pii`` /
``business_relevance`` onto the already-persisted row; the enriched-event
consumer then calls ``remediate_after_enrichment`` to apply the tenant's
configured action on that free-form signal. Storage + audit are stubbed so
these stay deterministic (no async audit queue / storage round-trip).
"""

import pytest

from core_api.services import governance_remediation
from core_api.services.organization_settings import ResolvedConfig

pytestmark = pytest.mark.asyncio


@pytest.fixture
def emitted(monkeypatch):
    """Capture governance audit emissions; returns the list of captured kwargs."""
    calls: list[dict] = []

    async def _record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(governance_remediation, "emit_governance_audit", _record)
    return calls


@pytest.fixture
def storage(monkeypatch):
    """Stub the storage client; record soft-delete / update calls."""
    actions: list[tuple] = []

    class _SC:
        async def soft_delete_memory(self, mid, tenant_id):
            actions.append(("soft_delete", mid, tenant_id))

        async def update_memory(self, mid, tenant_id, patch):
            actions.append(("update", mid, tenant_id, patch))

        async def purge_entity_artifacts(self, tenant_id, memory_id):
            actions.append(("purge_entities", memory_id, tenant_id))
            return {"links": 0, "relations": 0, "entities": 0}

    monkeypatch.setattr(governance_remediation, "get_storage_client", lambda: _SC())
    return actions


def _cfg(*, pii: dict | None = None, nb: dict | None = None) -> ResolvedConfig:
    gov: dict = {}
    if pii is not None:
        gov["pii"] = pii
    if nb is not None:
        gov["non_business"] = nb
    return ResolvedConfig({"governance": gov})


def _mem(**kw) -> dict:
    return {
        "id": kw.get("id", "m1"),
        "tenant_id": "t1",
        "agent_id": "a1",
        "content": kw.get("content", "free-form detail"),
        "metadata": kw.get("metadata", {}),
    }


async def test_disabled_is_noop(emitted, storage):
    outcome = await governance_remediation.remediate_after_enrichment(_mem(), _cfg())
    assert outcome.dropped is False
    assert emitted == []
    assert storage == []


async def test_missing_id_skips_without_side_effects(emitted, storage):
    # A malformed enriched-event payload without an id must not soft-delete the
    # literal "None" or stamp resource_id="None" on an audit row.
    cfg = _cfg(pii={"enabled": True, "action": "drop"})
    mem = _mem(id=None, metadata={"contains_pii": True})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is False
    assert storage == []
    assert emitted == []


async def test_pii_drop_soft_deletes_and_audits(emitted, storage):
    cfg = _cfg(pii={"enabled": True, "action": "drop"})
    mem = _mem(metadata={"contains_pii": True, "pii_types": ["health"]})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is True
    assert ("soft_delete", "m1", "t1") in storage
    assert any(c["action"] == "pii_drop" for c in emitted)


# ---------------------------------------------------------------------------
# H-02 — entity rows mined out of dropped content
# ---------------------------------------------------------------------------
#
# #808 named this case when it fixed the inline path: "entities mined out of
# dropped content are the same leak in another table". Both non-inline paths
# schedule extraction at write time, racing the verdict, and a soft-deleted
# memory still satisfies the link/relation foreign keys — so the names survived,
# listable tenant-wide, with nothing tying them to the drop.


async def test_a_pii_drop_purges_the_graph_rows(emitted, storage):
    cfg = _cfg(pii={"enabled": True, "action": "drop"})
    mem = _mem(metadata={"contains_pii": True, "pii_types": ["health"]})

    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)

    assert outcome.dropped is True
    assert ("purge_entities", "m1", "t1") in storage, storage


async def test_a_nonbusiness_drop_purges_the_graph_rows(emitted, storage):
    """Both destructive dispositions, not just one — separate branches, separate
    configs, and a tenant on either policy leaks identically without this."""
    cfg = _cfg(nb={"enabled": True, "disposition": "drop"})
    mem = _mem(metadata={"business_relevance": "personal"})

    await governance_remediation.remediate_after_enrichment(mem, cfg)

    assert ("purge_entities", "m1", "t1") in storage, storage


async def test_a_failed_purge_does_not_abort_the_handler(
    emitted, storage, monkeypatch, caplog
):
    """A raise here would nack the Pub/Sub event and redeliver it forever.

    ``consumer.handle_memory_enriched`` has no guard around remediation, and the
    dispatcher nacks on a handler exception. Redelivery re-runs the whole drop
    branch, emitting a SECOND ``critical=True`` audit for a memory that was
    already dropped — duplicate destructive entries in a tamper-evident log, for
    a row nothing further can be done to.

    The trade is bounded the other way: the memory is already gone, so the
    content is not live. What remains is graph rows, and the ERROR names the
    memory so they can be purged by hand.
    """

    class _SC:
        async def soft_delete_memory(self, mid, tenant_id):
            storage.append(("soft_delete", mid, tenant_id))

        async def purge_entity_artifacts(self, tenant_id, memory_id):
            raise RuntimeError("storage refused the purge")

    monkeypatch.setattr(governance_remediation, "get_storage_client", lambda: _SC())

    cfg = _cfg(nb={"enabled": True, "disposition": "drop"})
    mem = _mem(metadata={"business_relevance": "personal"})

    with caplog.at_level("ERROR"):
        outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)

    # The parent's verdict still reaches the caller.
    assert outcome.dropped is True
    assert ("soft_delete", "m1", "t1") in storage
    # And the failure is not silent.
    assert any("were NOT" in r.getMessage() for r in caplog.records), [
        r.getMessage()[:80] for r in caplog.records
    ]


async def test_a_non_destructive_verdict_purges_nothing(emitted, storage):
    """OVER-REFUSAL GUARD. ``flag`` and ``keep_private`` leave the row readable.

    The graph rows describe content that is still there and still allowed, so
    removing them would destroy data no policy asked to remove.
    """
    cfg = _cfg(
        pii={"enabled": True, "action": "flag"},
        nb={"enabled": True, "disposition": "keep_private"},
    )
    mem = _mem(metadata={"contains_pii": True, "business_relevance": "personal"})

    await governance_remediation.remediate_after_enrichment(mem, cfg)

    assert not any(kind == "purge_entities" for kind, *_ in storage), storage


async def test_pii_mask_config_flags_but_records_intent(emitted, storage):
    # Fast mode can't redact a free-form LLM span; a mask policy flags the row
    # but stays distinguishable from a genuine flag policy in the audit.
    cfg = _cfg(pii={"enabled": True, "action": "mask"})
    mem = _mem(metadata={"contains_pii": True, "pii_types": ["health"]})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is False
    assert storage == []  # nothing redacted/dropped
    flag = next(c for c in emitted if c["action"] == "pii_flag")
    assert flag["detail"]["configured_action"] == "pii_mask"


async def test_pii_flag_config_records_no_configured_action(emitted, storage):
    cfg = _cfg(pii={"enabled": True, "action": "flag"})
    mem = _mem(metadata={"contains_pii": True})
    await governance_remediation.remediate_after_enrichment(mem, cfg)
    flag = next(c for c in emitted if c["action"] == "pii_flag")
    assert "configured_action" not in flag["detail"]


async def test_nonbusiness_keep_private_updates_visibility(emitted, storage):
    cfg = _cfg(nb={"enabled": True, "disposition": "keep_private"})
    mem = _mem(metadata={"business_relevance": "personal"})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is False
    assert ("update", "m1", "t1", {"visibility": "scope_agent"}) in storage
    assert any(c["action"] == "nonbusiness_keep_private" for c in emitted)


async def test_nonbusiness_drop_soft_deletes(emitted, storage):
    cfg = _cfg(nb={"enabled": True, "disposition": "drop"})
    mem = _mem(metadata={"business_relevance": "personal"})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is True
    assert ("soft_delete", "m1", "t1") in storage
    assert any(c["action"] == "nonbusiness_drop" for c in emitted)


async def test_business_content_is_noop(emitted, storage):
    cfg = _cfg(nb={"enabled": True, "disposition": "drop"})
    mem = _mem(metadata={"business_relevance": "business"})
    outcome = await governance_remediation.remediate_after_enrichment(mem, cfg)
    assert outcome.dropped is False
    assert storage == []
    assert emitted == []


@pytest.mark.parametrize(
    ("cfg_kwargs", "metadata", "drop_action"),
    [
        (
            {"pii": {"enabled": True, "action": "drop"}},
            {"contains_pii": True},
            "pii_drop",
        ),
        (
            {"nb": {"enabled": True, "disposition": "drop"}},
            {"business_relevance": "personal"},
            "nonbusiness_drop",
        ),
    ],
)
async def test_drop_audits_before_soft_delete(
    monkeypatch, cfg_kwargs, metadata, drop_action
):
    # Compliance invariant: the audit must be recorded BEFORE the destructive
    # soft-delete, so a delete that succeeds before a failing audit can't leave
    # an untracked deletion in the tamper-evident log (mirrors the audit-before-
    # mutate ordering in GovernanceScanContent). Capture both into one ordered log.
    order: list[str] = []

    async def _audit(*_a, **kw):
        order.append(f"audit:{kw['action']}")

    class _SC:
        async def soft_delete_memory(self, _mid, _tenant_id):
            order.append("soft_delete")

        async def purge_entity_artifacts(self, _tenant_id, _memory_id):
            order.append("purge_entities")
            return {"links": 0, "relations": 0, "entities": 0}

    monkeypatch.setattr(governance_remediation, "emit_governance_audit", _audit)
    monkeypatch.setattr(governance_remediation, "get_storage_client", lambda: _SC())

    outcome = await governance_remediation.remediate_after_enrichment(
        _mem(metadata=metadata), _cfg(**cfg_kwargs)
    )
    assert outcome.dropped is True
    # The graph purge comes LAST, and that is also an ordering invariant rather
    # than an accident: purging before the soft-delete would destroy graph rows
    # for a memory that is still live if the delete then failed. Nothing else
    # would put those rows back.
    assert order == [f"audit:{drop_action}", "soft_delete", "purge_entities"]
