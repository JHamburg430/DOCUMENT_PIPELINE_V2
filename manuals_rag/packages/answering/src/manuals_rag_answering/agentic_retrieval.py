from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from manuals_rag_common.config import settings
from manuals_rag_common.ollama import chat_json
from manuals_rag_retrieval.query_analysis import analyze_query
from manuals_rag_retrieval.retriever import (
    assess_evidence_sufficiency,
    assemble_agent_context,
    retrieve_with_strategy,
)
from manuals_rag_schemas.documents import SearchResult


RetrievalStrategy = Literal["hybrid", "broad", "dense", "sparse", "structural"]
PlanMode = Literal["single", "parallel", "dependent"]


class RetrievalHop(BaseModel):
    hop_id: str
    objective: str
    query: str
    strategy: RetrievalStrategy = "hybrid"
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    recovery_for: str | None = None


class RetrievalPlan(BaseModel):
    mode: PlanMode = "single"
    rationale: str = ""
    hops: list[RetrievalHop]


class AgenticState(TypedDict, total=False):
    query: str
    corpus_ids: list[str]
    filters: dict[str, object]
    max_hops: int
    plan: dict[str, Any]
    pending_hop_ids: list[str]
    completed_hop_ids: list[str]
    hop_results: dict[str, list[dict[str, Any]]]
    evidence_ledger: dict[str, dict[str, Any]]
    retrieval_results: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    stop_reason: str
    sufficient: bool
    started_at: float
    duration_ms: float


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["single", "parallel", "dependent"]},
        "rationale": {"type": "string"},
        "hops": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "hop_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "query": {"type": "string"},
                    "strategy": {
                        "type": "string",
                        "enum": ["hybrid", "broad", "dense", "sparse", "structural"],
                    },
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "required": {"type": "boolean"},
                },
                "required": ["hop_id", "objective", "query", "strategy", "depends_on", "required"],
            },
        },
    },
    "required": ["mode", "rationale", "hops"],
}


REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


PLANNER_PROMPT = """
You plan bounded retrieval over technical manuals. Return only JSON matching the schema.
Decompose only when separate evidence is genuinely required. Use:
- single: one independently answerable lookup;
- parallel: multiple independent facts or documents must all be retrieved;
- dependent: a later lookup needs an entity, value, component, or location discovered earlier.
Each hop must be a standalone search query without pronouns. Keep at most four hops.
Choose structural for tables, settings, specifications, procedures, or troubleshooting rows;
sparse for exact identifiers and quoted phrases; dense for conceptual language; broad when the
request is exploratory; otherwise hybrid. Dependencies must reference earlier hop_id values.
Never include an answer or facts not present in the user's question.
""".strip()


LLAMAINDEX_PLANNER_PROMPT = """
You are a LlamaIndex-style subquestion planner for technical-manual research. Return only JSON.
Treat every retrieval strategy as a query-engine tool:
- structural: exact tables, specifications, settings, procedures, and troubleshooting rows;
- sparse: identifiers, model numbers, alarm codes, and exact phrases;
- dense: conceptual or paraphrased requests;
- hybrid: mixed exact and semantic evidence;
- broad: exploratory discovery when the target is not known yet.
Create independently answerable subquestions. Use parallel mode when subquestions can run alone and
dependent mode only when a later subquestion must incorporate an entity discovered earlier. Prefer
multiple focused subquestions over one compound query, but keep the plan within four hops. Dependencies
must reference earlier hop_id values. Do not provide the answer or invent manual facts.
""".strip()


def _parallel_scope_plan(query: str) -> RetrievalPlan | None:
    """Recognize a common, document-general comparison shape without an LLM."""
    match = re.match(
        r"^\s*for\s+(?P<scopes>.+?),\s*(?P<request>(?:what|which|how|where|when)\b.+)$",
        query,
        flags=re.I,
    )
    if not match:
        return None
    scopes = [part.strip(" ,.;?") for part in re.split(r"\s+and\s+", match.group("scopes"), flags=re.I)]
    if not 2 <= len(scopes) <= 4 or any(len(scope.split()) > 8 for scope in scopes):
        return None

    request = match.group("request").strip()
    detail_match = re.match(r"(?P<prefix>.+?\bfor\s+)(?P<details>.+?)\??$", request, flags=re.I)
    details: list[str] = []
    if detail_match:
        details = [
            part.strip(" ,.;?")
            for part in re.split(r"\s+and\s+", detail_match.group("details"), flags=re.I)
        ]
    pair_details = len(details) == len(scopes)

    hops: list[RetrievalHop] = []
    for index, scope in enumerate(scopes):
        if pair_details and detail_match:
            hop_query = f"For {scope}, {detail_match.group('prefix')}{details[index]}?"
        else:
            hop_query = f"{query.rstrip('?')} Focus only on {scope}."
        hops.append(
            RetrievalHop(
                hop_id=f"side_{index + 1}",
                objective=f"Retrieve the requested evidence for {scope}",
                query=hop_query,
                strategy="structural",
            )
        )
    return RetrievalPlan(
        mode="parallel",
        rationale="The request asks for independently retrievable evidence across multiple named scopes.",
        hops=hops,
    )


def _troubleshooting_facet_plan(query: str) -> RetrievalPlan | None:
    match = re.match(
        r"^\s*what\s+causes?\s+(?P<target>.+?)(?:,)?\s+and\s+how\s+should\s+(?:it|this|that)\s+be\s+corrected\??$",
        query,
        flags=re.I,
    )
    if not match:
        return None
    target = match.group("target").strip(" ,.;?")
    return RetrievalPlan(
        mode="parallel",
        rationale="The request requires separate cause and corrective-action evidence.",
        hops=[
            RetrievalHop(
                hop_id="cause",
                objective=f"Find the documented cause of {target}",
                query=f"What causes {target}?",
                strategy="structural",
            ),
            RetrievalHop(
                hop_id="corrective_action",
                objective=f"Find the documented corrective action for {target}",
                query=f"How should {target} be corrected?",
                strategy="structural",
            ),
        ],
    )


def _heuristic_plan(query: str) -> RetrievalPlan:
    troubleshooting_plan = _troubleshooting_facet_plan(query)
    if troubleshooting_plan is not None:
        return troubleshooting_plan
    scoped_plan = _parallel_scope_plan(query)
    if scoped_plan is not None:
        return scoped_plan
    analysis = analyze_query(query)
    identifiers = list(dict.fromkeys(analysis.product_identifiers))
    if "comparison" in analysis.query_types and len(identifiers) >= 2:
        hops = [
            RetrievalHop(
                hop_id=f"side_{index + 1}",
                objective=f"Retrieve the requested evidence for {identifier}",
                query=f"{query}\nFocus on the requested evidence for {identifier}.",
                strategy="structural",
            )
            for index, identifier in enumerate(identifiers[:4])
        ]
        return RetrievalPlan(mode="parallel", rationale="Explicit comparison across identifiers.", hops=hops)

    clauses = [part.strip(" ,.;") for part in re.split(r"\b(?:then|after that|using that)\b", query, flags=re.I) if part.strip()]
    if len(clauses) >= 2:
        hops: list[RetrievalHop] = []
        for index, clause in enumerate(clauses[:4]):
            hops.append(
                RetrievalHop(
                    hop_id=f"hop_{index + 1}",
                    objective=clause,
                    query=clause,
                    strategy="hybrid",
                    depends_on=[] if index == 0 else [f"hop_{index}"],
                )
            )
        return RetrievalPlan(mode="dependent", rationale="The request contains an explicit dependency sequence.", hops=hops)

    strategy: RetrievalStrategy = "structural" if set(analysis.query_types).intersection(
        {"configuration", "specification", "troubleshooting", "how_to"}
    ) else "hybrid"
    return RetrievalPlan(
        mode="single",
        rationale="One independently answerable retrieval objective.",
        hops=[RetrievalHop(hop_id="hop_1", objective=query, query=query, strategy=strategy)],
    )


def plan_retrieval(query: str, *, use_llm: bool = True) -> RetrievalPlan:
    if not use_llm:
        return _heuristic_plan(query)
    try:
        payload, _raw = chat_json(
            model=settings.ollama_fast_model,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": query},
            ],
            json_schema=PLAN_SCHEMA,
            think=False,
            timeout=45.0,
            num_predict=700,
            purpose="agentic_retrieval_plan",
        )
        plan = RetrievalPlan.model_validate(payload)
        _validate_plan(plan)
        return plan
    except Exception:
        return _heuristic_plan(query)


def _llamaindex_heuristic_plan(query: str) -> RetrievalPlan:
    """Subquestion-oriented fallback that is intentionally independent of LangGraph planning."""
    base = _heuristic_plan(query)
    hops: list[RetrievalHop] = []
    for index, hop in enumerate(base.hops, start=1):
        strategy = hop.strategy
        analysis = analyze_query(hop.query)
        if analysis.product_identifiers:
            strategy = "sparse"
        elif set(analysis.query_types).intersection({"configuration", "specification", "troubleshooting", "how_to"}):
            strategy = "structural"
        hops.append(
            hop.model_copy(
                update={
                    "hop_id": f"subquestion_{index}",
                    "strategy": strategy,
                    "depends_on": [] if not hop.depends_on else [f"subquestion_{index - 1}"],
                }
            )
        )
    return RetrievalPlan(
        mode=base.mode,
        rationale=f"LlamaIndex subquestion decomposition: {base.rationale}",
        hops=hops,
    )


def plan_llamaindex_retrieval(query: str, *, use_llm: bool = True) -> RetrievalPlan:
    if not use_llm:
        return _llamaindex_heuristic_plan(query)
    try:
        payload, _raw = chat_json(
            model=settings.ollama_fast_model,
            messages=[
                {"role": "system", "content": LLAMAINDEX_PLANNER_PROMPT},
                {"role": "user", "content": query},
            ],
            json_schema=PLAN_SCHEMA,
            think=False,
            timeout=45.0,
            num_predict=700,
            purpose="llamaindex_subquestion_plan",
        )
        plan = RetrievalPlan.model_validate(payload)
        _validate_plan(plan)
        return plan
    except Exception:
        return _llamaindex_heuristic_plan(query)


def _validate_plan(plan: RetrievalPlan) -> None:
    if not plan.hops:
        raise ValueError("Retrieval plan must contain at least one hop.")
    ids = [hop.hop_id for hop in plan.hops]
    if len(ids) != len(set(ids)):
        raise ValueError("Retrieval hop IDs must be unique.")
    seen: set[str] = set()
    for hop in plan.hops:
        if not hop.query.strip() or not hop.objective.strip():
            raise ValueError("Retrieval hops require an objective and query.")
        if any(dependency not in seen for dependency in hop.depends_on):
            raise ValueError("Retrieval dependencies must reference earlier hops.")
        seen.add(hop.hop_id)


def _evidence_excerpt(results: list[SearchResult], *, max_chars: int = 2400) -> str:
    parts: list[str] = []
    used = 0
    for result in results[:4]:
        item = (
            f"Document: {result.title} ({result.source_document_id}); "
            f"section: {' > '.join(result.section_path)}; evidence: {result.content}"
        )
        item = item[:700]
        if used + len(item) > max_chars:
            item = item[: max(0, max_chars - used)]
        if item:
            parts.append(item)
            used += len(item)
        if used >= max_chars:
            break
    return "\n".join(parts)


def _dependency_anchors(results: list[SearchResult]) -> list[str]:
    anchors: list[str] = []
    compact_anchors: set[str] = set()
    for result in results[:6]:
        candidates = [str(value) for value in result.metadata.get("identifier_tokens", [])]
        candidates.extend(
            re.findall(r"\b[A-Z]{1,8}(?:[-:/][A-Z0-9]{1,12})+\b", result.content, flags=re.I)
        )
        for candidate in candidates:
            normalized = candidate.strip(" ,.;:()[]")
            # Dependency anchors should be concrete technical identifiers, not
            # arbitrary hyphenated prose such as "bus-powered" or "and/or".
            if not normalized or not any(character.isdigit() for character in normalized):
                continue
            compact = re.sub(r"[^a-z0-9]", "", normalized.lower())
            if compact and compact not in compact_anchors:
                anchors.append(normalized)
                compact_anchors.add(compact)
            if len(anchors) >= 12:
                return anchors
    return anchors


def refine_dependent_query(hop: RetrievalHop, dependency_results: list[SearchResult], *, use_llm: bool = True) -> str:
    if not dependency_results:
        return hop.query
    evidence = _evidence_excerpt(dependency_results)
    anchors = _dependency_anchors(dependency_results)
    fallback = hop.query
    if anchors:
        fallback = f"{hop.objective}. Relevant prior-hop identifiers: {', '.join(anchors[:6])}"
    else:
        fallback = f"{fallback}\nRelevant prior-hop evidence: {evidence}"
    if not use_llm:
        return fallback
    try:
        payload, _raw = chat_json(
            model=settings.ollama_fast_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the pending technical-manual search query using only concrete entities or values "
                        "supported by the prior-hop evidence. Return one standalone query. Do not answer it."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Objective: {hop.objective}\nOriginal query: {hop.query}\nPrior evidence:\n{evidence}",
                },
            ],
            json_schema=REFINE_SCHEMA,
            think=False,
            timeout=45.0,
            num_predict=240,
            purpose="agentic_retrieval_refine",
        )
        refined = str(payload.get("query") or "").strip()
        if anchors and not any(anchor.lower() in refined.lower() for anchor in anchors):
            return fallback
        return refined or fallback
    except Exception:
        return fallback


def refine_llamaindex_subquestion(
    hop: RetrievalHop,
    dependency_results: list[SearchResult],
    *,
    use_llm: bool = True,
) -> str:
    """Apply a LlamaIndex-style query transformation to a dependent subquestion."""
    if not dependency_results:
        return hop.query
    anchors = _dependency_anchors(dependency_results)
    evidence = _evidence_excerpt(dependency_results)
    fallback = (
        f"{hop.query.rstrip(' ?')}; constrain the lookup to {', '.join(anchors[:6])}"
        if anchors
        else f"{hop.query}\nSubquestion context: {evidence}"
    )
    if not use_llm:
        return fallback
    try:
        payload, _raw = chat_json(
            model=settings.ollama_fast_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Transform the unresolved subquestion into one standalone query-engine query. "
                        "Bind it to concrete entities supported by the dependency evidence, preserve the "
                        "requested answer facet, and do not answer the query."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Subquestion: {hop.query}\nObjective: {hop.objective}\nDependency evidence:\n{evidence}",
                },
            ],
            json_schema=REFINE_SCHEMA,
            think=False,
            timeout=45.0,
            num_predict=240,
            purpose="llamaindex_query_transform",
        )
        transformed = str(payload.get("query") or "").strip()
        if anchors and not any(anchor.lower() in transformed.lower() for anchor in anchors):
            return fallback
        return transformed or fallback
    except Exception:
        return fallback


def _plan_hops(state: AgenticState) -> list[RetrievalHop]:
    return RetrievalPlan.model_validate(state["plan"]).hops


def _hop_by_id(state: AgenticState, hop_id: str) -> RetrievalHop:
    return next(hop for hop in _plan_hops(state) if hop.hop_id == hop_id)


def _results_for_ids(state: AgenticState, hop_ids: list[str]) -> list[SearchResult]:
    results: list[SearchResult] = []
    plan = RetrievalPlan.model_validate(state["plan"])
    ledger = state.get("evidence_ledger", {})
    for hop_id in hop_ids:
        recovery_ids = [
            hop.hop_id
            for hop in plan.hops
            if hop.recovery_for == hop_id and bool(ledger.get(hop.hop_id, {}).get("sufficient"))
        ]
        # A successful recovery is the best dependency evidence for the next hop.
        # Retain the original result set as additional grounding when it exists.
        for result_hop_id in [*recovery_ids, hop_id]:
            results.extend(
                SearchResult.model_validate(item)
                for item in state.get("hop_results", {}).get(result_hop_id, [])
            )
    return results


def _assess_hop_evidence(
    query: str,
    results: list[SearchResult],
    *,
    dependency_anchors: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Narrow the global query sufficiency contract to the hop's explicit objective."""
    base = assess_evidence_sufficiency(query, results)
    payload = base.to_dict()
    if not results:
        return False, payload
    lowered_query = query.lower()
    evidence = "\n".join(result.content for result in results).lower()
    anchors = [anchor for anchor in (dependency_anchors or []) if anchor]
    anchored_results = [
        result
        for result in results
        if any(
            re.sub(r"[^a-z0-9]", "", anchor.lower())
            in re.sub(r"[^a-z0-9]", "", result.content.lower())
            for anchor in anchors
        )
    ]
    scoped_evidence = "\n".join(result.content for result in anchored_results).lower() if anchors else evidence
    explicit_checks: list[bool] = []
    if re.search(r"\b(?:cause|why|reason|due to)\b", lowered_query):
        explicit_checks.append(bool(re.search(r"\b(?:cause|because|due to|results? from|occurs? when|if)\b", scoped_evidence)))
    if re.search(r"\b(?:corrective action|remedy|resolve|fix)\b", lowered_query):
        explicit_checks.append(
            bool(re.search(r"\b(?:correct|remedy|resolve|fix|replace|reconnect|restart|check|set|adjust|recalibrat\w*|remove|install|ensure|verify)\b", scoped_evidence))
        )
    if re.search(r"\b(?:where|menu|screen|tab|section|page)\b", lowered_query):
        explicit_checks.append(
            max((len(result.section_path) for result in results), default=0) >= 2
            or bool(re.search(r"\b(?:menu|screen|tab|section|page|under|within)\b", scoped_evidence))
        )
    if re.search(r"\b(?:value|maximum|minimum|range|tolerance|voltage|current|temperature|distance|time)\b", lowered_query):
        explicit_checks.append(bool(re.search(r"\b\d+(?:\.\d+)?\b", scoped_evidence)))
    if re.search(r"\b(?:orientation|straight|right[- ]?angle|angled)\b", lowered_query):
        explicit_checks.append(bool(re.search(r"\b(?:straight|right[- ]?angle|angled|vertical|horizontal)\b", scoped_evidence)))
    if re.search(r"\b(?:model|part number|catalog(?:ue)? number)\b", lowered_query):
        explicit_checks.append(bool(re.search(r"\b(?=[a-z0-9:/-]*\d)[a-z][a-z0-9]*(?:[-:/][a-z0-9]+)+\b", scoped_evidence)))
    coverage_sufficient = (
        all(explicit_checks) and payload["query_term_coverage"] >= 0.25
        if explicit_checks
        else payload["query_term_coverage"] >= 0.4
    )
    sufficient = coverage_sufficient and (not anchors or bool(anchored_results))
    payload["global_sufficient"] = payload["sufficient"]
    payload["sufficient"] = sufficient
    payload["scope"] = "hop_objective"
    payload["dependency_anchors"] = anchors
    payload["anchored_chunk_ids"] = [result.chunk_id for result in anchored_results]
    return sufficient, payload


class AgenticRetrievalController:
    def __init__(
        self,
        *,
        use_llm: bool = True,
        planner: Callable[[str], RetrievalPlan] | None = None,
        refiner: Callable[[RetrievalHop, list[SearchResult]], str] | None = None,
        retriever: Callable[[str, list[str], dict[str, object], RetrievalStrategy, int], list[SearchResult]] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.use_llm = use_llm
        self.planner = planner or (lambda query: plan_retrieval(query, use_llm=use_llm))
        self.refiner = refiner or (
            lambda hop, results: refine_dependent_query(hop, results, use_llm=use_llm)
        )
        self.retriever = retriever or (
            lambda query, corpus_ids, filters, strategy, limit: retrieve_with_strategy(
                query, corpus_ids, filters, strategy=strategy, limit=limit
            )
        )
        self.event_callback = event_callback

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_callback is not None:
            self.event_callback({"event": event, "policy": "langgraph_state_graph", **payload})

    def initialize(self, state: AgenticState) -> AgenticState:
        started_at = float(state.get("started_at") or perf_counter())
        plan = self.planner(state["query"])
        _validate_plan(plan)
        max_hops = max(1, min(int(state.get("max_hops", 4)), 8))
        self._emit("plan_completed", plan=plan.model_dump(), max_hops=max_hops)
        return {
            **state,
            "started_at": started_at,
            "max_hops": max_hops,
            "plan": plan.model_dump(),
            "pending_hop_ids": [hop.hop_id for hop in plan.hops],
            "completed_hop_ids": [],
            "hop_results": {},
            "evidence_ledger": {},
            "sufficient": False,
            "stop_reason": "",
        }

    def execute_next(self, state: AgenticState) -> AgenticState:
        pending = list(state.get("pending_hop_ids", []))
        completed = list(state.get("completed_hop_ids", []))
        runnable = next(
            (
                hop_id
                for hop_id in pending
                if all(dependency in completed for dependency in _hop_by_id(state, hop_id).depends_on)
            ),
            None,
        )
        if runnable is None:
            self._emit("agent_stopped", stop_reason="no_runnable_hop")
            return {**state, "stop_reason": "no_runnable_hop"}

        hop = _hop_by_id(state, runnable)
        dependency_results = _results_for_ids(state, hop.depends_on)
        dependency_anchors = _dependency_anchors(dependency_results)
        if dependency_anchors and hop.depends_on:
            dependency_queries = " ".join(_hop_by_id(state, hop_id).query for hop_id in hop.depends_on)
            compact_dependency_queries = re.sub(r"[^a-z0-9]", "", dependency_queries.lower())
            novel_anchors = [
                anchor
                for anchor in dependency_anchors
                if re.sub(r"[^a-z0-9]", "", anchor.lower()) not in compact_dependency_queries
            ]
            if novel_anchors:
                dependency_anchors = novel_anchors
        executed_query = self.refiner(hop, dependency_results) if hop.depends_on else hop.query
        executed_strategy: RetrievalStrategy = hop.strategy
        if dependency_anchors and hop.strategy == "hybrid":
            executed_query = (
                f"{hop.objective}. Relevant prior-hop identifiers: {', '.join(dependency_anchors[:6])}"
            )
            executed_strategy = "sparse"
        self._emit(
            "hop_started",
            hop_id=hop.hop_id,
            objective=hop.objective,
            planned_query=hop.query,
            executed_query=executed_query,
            planned_strategy=hop.strategy,
            strategy=executed_strategy,
            depends_on=hop.depends_on,
            dependency_anchors=dependency_anchors,
            recovery_for=hop.recovery_for,
        )
        results = self.retriever(
            executed_query,
            state["corpus_ids"],
            state.get("filters", {}),
            executed_strategy,
            10,
        )
        if hop.recovery_for and not dependency_anchors:
            original_assessment = state.get("evidence_ledger", {}).get(hop.recovery_for, {}).get("assessment", {})
            dependency_anchors = list(original_assessment.get("dependency_anchors") or [])
            if not dependency_anchors:
                dependency_anchors = [
                    match
                    for match in re.findall(r"\b[A-Z]{1,8}(?:[-:/][A-Z0-9]{1,12})+\b", executed_query, flags=re.I)
                    if any(character.isdigit() for character in match)
                ]
        hop_sufficient, assessment = _assess_hop_evidence(
            hop.objective,
            results,
            dependency_anchors=dependency_anchors,
        )
        hop_results = dict(state.get("hop_results", {}))
        hop_results[runnable] = [result.model_dump() for result in results]
        ledger = dict(state.get("evidence_ledger", {}))
        ledger[runnable] = {
            "objective": hop.objective,
            "planned_query": hop.query,
            "executed_query": executed_query,
            "planned_strategy": hop.strategy,
            "strategy": executed_strategy,
            "depends_on": hop.depends_on,
            "required": hop.required,
            "recovery_for": hop.recovery_for,
            "sufficient": hop_sufficient,
            "assessment": assessment,
            "chunk_ids": [result.chunk_id for result in results],
            "document_ids": sorted({result.source_document_id for result in results}),
        }
        self._emit(
            "hop_completed",
            hop_id=runnable,
            sufficient=hop_sufficient,
            assessment=assessment,
            result_count=len(results),
            results=[result.model_dump() for result in results],
            ledger_entry=ledger[runnable],
        )
        pending.remove(runnable)
        completed.append(runnable)
        return {
            **state,
            "pending_hop_ids": pending,
            "completed_hop_ids": completed,
            "hop_results": hop_results,
            "evidence_ledger": ledger,
        }

    def judge(self, state: AgenticState) -> AgenticState:
        plan = RetrievalPlan.model_validate(state["plan"])
        completed = list(state.get("completed_hop_ids", []))
        pending = list(state.get("pending_hop_ids", []))
        ledger = dict(state.get("evidence_ledger", {}))
        required = [hop for hop in plan.hops if hop.required and hop.recovery_for is None]

        for hop in required:
            item = ledger.get(hop.hop_id)
            already_recovered = any(candidate.recovery_for == hop.hop_id for candidate in plan.hops)
            if (
                item
                and not item.get("sufficient")
                and not already_recovered
                and len(completed) + len(pending) < state["max_hops"]
            ):
                recovery = RetrievalHop(
                    hop_id=f"{hop.hop_id}_recovery",
                    objective=hop.objective,
                    query=str(item.get("executed_query") or hop.query),
                    strategy="broad",
                    depends_on=[],
                    required=False,
                    recovery_for=hop.hop_id,
                )
                plan.hops.append(recovery)
                pending.insert(0, recovery.hop_id)
                self._emit(
                    "recovery_scheduled",
                    hop_id=recovery.hop_id,
                    recovery_for=hop.hop_id,
                    query=recovery.query,
                    strategy=recovery.strategy,
                )
                return {**state, "plan": plan.model_dump(), "pending_hop_ids": pending}

        sufficient_ids: set[str] = {
            hop_id for hop_id, item in ledger.items() if bool(item.get("sufficient"))
        }
        for hop in plan.hops:
            if hop.recovery_for and hop.hop_id in sufficient_ids:
                sufficient_ids.add(hop.recovery_for)
        all_required_sufficient = all(hop.hop_id in sufficient_ids for hop in required)
        exhausted = len(completed) >= state["max_hops"]
        blocked = bool(state.get("stop_reason"))
        done = all_required_sufficient or exhausted or blocked or not pending
        if not done:
            return {**state, "plan": plan.model_dump(), "pending_hop_ids": pending}

        hop_result_sets = {
            hop_id: [SearchResult.model_validate(item) for item in items]
            for hop_id, items in state.get("hop_results", {}).items()
        }
        final_results = assemble_agent_context(state["query"], hop_result_sets, limit=10)
        stop_reason = (
            "sufficient" if all_required_sufficient else "hop_budget_exhausted" if exhausted else state.get("stop_reason") or "plan_exhausted"
        )
        duration_ms = round((perf_counter() - float(state["started_at"])) * 1000, 2)
        trace = {
            "policy": "langgraph_state_graph",
            "mode": plan.mode,
            "rationale": plan.rationale,
            "plan": plan.model_dump(),
            "max_hops": state["max_hops"],
            "completed_hops": completed,
            "pending_hops": pending,
            "evidence_ledger": ledger,
            "sufficient": all_required_sufficient,
            "stop_reason": stop_reason,
            "duration_ms": duration_ms,
            "cost": {
                "retrieval_calls": len(completed),
                "llm_token_estimate": sum(len(str(item.get("executed_query") or "")) for item in ledger.values()) // 4,
            },
        }
        self._emit(
            "retrieval_completed",
            sufficient=all_required_sufficient,
            stop_reason=stop_reason,
            duration_ms=duration_ms,
            result_count=len(final_results),
            results=[result.model_dump() for result in final_results],
            trace=trace,
        )
        return {
            **state,
            "plan": plan.model_dump(),
            "pending_hop_ids": pending,
            "retrieval_results": [result.model_dump() for result in final_results],
            "retrieval_trace": trace,
            "sufficient": all_required_sufficient,
            "stop_reason": stop_reason,
            "duration_ms": duration_ms,
        }

    @staticmethod
    def should_continue(state: AgenticState) -> str:
        return "done" if state.get("retrieval_results") is not None else "continue"


class LlamaIndexAgenticController:
    """Independent subquestion/query-engine policy used by the LlamaIndex workflow."""

    def __init__(
        self,
        *,
        use_llm: bool = True,
        planner: Callable[[str], RetrievalPlan] | None = None,
        transformer: Callable[[RetrievalHop, list[SearchResult]], str] | None = None,
        retriever: Callable[[str, list[str], dict[str, object], RetrievalStrategy, int], list[SearchResult]] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.use_llm = use_llm
        self.planner = planner or (lambda query: plan_llamaindex_retrieval(query, use_llm=use_llm))
        self.transformer = transformer or (
            lambda hop, results: refine_llamaindex_subquestion(hop, results, use_llm=use_llm)
        )
        self.retriever = retriever or (
            lambda query, corpus_ids, filters, strategy, limit: retrieve_with_strategy(
                query, corpus_ids, filters, strategy=strategy, limit=limit
            )
        )
        self.event_callback = event_callback

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_callback is not None:
            self.event_callback({"event": event, "policy": "llamaindex_subquestion", **payload})

    def initialize(self, state: AgenticState) -> AgenticState:
        started_at = float(state.get("started_at") or perf_counter())
        plan = self.planner(state["query"])
        _validate_plan(plan)
        max_hops = max(1, min(int(state.get("max_hops", 4)), 8))
        self._emit("plan_completed", plan=plan.model_dump(), max_hops=max_hops)
        return {
            **state,
            "started_at": started_at,
            "max_hops": max_hops,
            "plan": plan.model_dump(),
            "pending_hop_ids": [hop.hop_id for hop in plan.hops],
            "completed_hop_ids": [],
            "hop_results": {},
            "evidence_ledger": {},
            "sufficient": False,
            "stop_reason": "",
        }

    @staticmethod
    def _route_tool(hop: RetrievalHop, dependency_anchors: list[str]) -> RetrievalStrategy:
        if hop.recovery_for:
            return hop.strategy
        if dependency_anchors:
            return "sparse"
        analysis = analyze_query(hop.query)
        if analysis.product_identifiers:
            return "sparse"
        if set(analysis.query_types).intersection({"configuration", "specification", "troubleshooting", "how_to"}):
            return "structural"
        return hop.strategy

    def execute_next(self, state: AgenticState) -> AgenticState:
        pending = list(state.get("pending_hop_ids", []))
        completed = list(state.get("completed_hop_ids", []))
        runnable = next(
            (
                hop_id
                for hop_id in pending
                if all(dependency in completed for dependency in _hop_by_id(state, hop_id).depends_on)
            ),
            None,
        )
        if runnable is None:
            self._emit("agent_stopped", stop_reason="no_runnable_subquestion")
            return {**state, "stop_reason": "no_runnable_subquestion"}

        hop = _hop_by_id(state, runnable)
        dependency_results = _results_for_ids(state, hop.depends_on)
        dependency_anchors = _dependency_anchors(dependency_results)
        executed_query = self.transformer(hop, dependency_results) if hop.depends_on else hop.query
        executed_strategy = self._route_tool(hop, dependency_anchors)
        self._emit(
            "tool_selected",
            hop_id=hop.hop_id,
            tool=executed_strategy,
            reason="dependency anchor routing" if dependency_anchors else "subquestion analysis",
        )
        self._emit(
            "hop_started",
            hop_id=hop.hop_id,
            objective=hop.objective,
            planned_query=hop.query,
            executed_query=executed_query,
            planned_strategy=hop.strategy,
            strategy=executed_strategy,
            depends_on=hop.depends_on,
            dependency_anchors=dependency_anchors,
            recovery_for=hop.recovery_for,
        )
        results = self.retriever(
            executed_query,
            state["corpus_ids"],
            state.get("filters", {}),
            executed_strategy,
            10,
        )
        if hop.recovery_for and not dependency_anchors:
            original = state.get("evidence_ledger", {}).get(hop.recovery_for, {})
            dependency_anchors = list((original.get("assessment") or {}).get("dependency_anchors") or [])
        hop_sufficient, assessment = _assess_hop_evidence(
            hop.objective,
            results,
            dependency_anchors=dependency_anchors,
        )
        hop_results = dict(state.get("hop_results", {}))
        hop_results[runnable] = [result.model_dump() for result in results]
        ledger = dict(state.get("evidence_ledger", {}))
        ledger[runnable] = {
            "objective": hop.objective,
            "planned_query": hop.query,
            "executed_query": executed_query,
            "planned_strategy": hop.strategy,
            "strategy": executed_strategy,
            "tool": executed_strategy,
            "depends_on": hop.depends_on,
            "required": hop.required,
            "recovery_for": hop.recovery_for,
            "sufficient": hop_sufficient,
            "assessment": assessment,
            "chunk_ids": [result.chunk_id for result in results],
            "document_ids": sorted({result.source_document_id for result in results}),
        }
        self._emit(
            "hop_completed",
            hop_id=runnable,
            sufficient=hop_sufficient,
            assessment=assessment,
            result_count=len(results),
            results=[result.model_dump() for result in results],
            ledger_entry=ledger[runnable],
        )
        pending.remove(runnable)
        completed.append(runnable)
        return {
            **state,
            "pending_hop_ids": pending,
            "completed_hop_ids": completed,
            "hop_results": hop_results,
            "evidence_ledger": ledger,
        }

    def judge(self, state: AgenticState) -> AgenticState:
        plan = RetrievalPlan.model_validate(state["plan"])
        completed = list(state.get("completed_hop_ids", []))
        pending = list(state.get("pending_hop_ids", []))
        ledger = dict(state.get("evidence_ledger", {}))
        required = [hop for hop in plan.hops if hop.required and hop.recovery_for is None]

        for hop in required:
            item = ledger.get(hop.hop_id)
            already_recovered = any(candidate.recovery_for == hop.hop_id for candidate in plan.hops)
            if (
                item
                and not item.get("sufficient")
                and not already_recovered
                and len(completed) + len(pending) < state["max_hops"]
            ):
                previous_tool = str(item.get("strategy") or hop.strategy)
                recovery_tool: RetrievalStrategy = {
                    "sparse": "dense",
                    "dense": "sparse",
                    "structural": "hybrid",
                    "hybrid": "broad",
                    "broad": "structural",
                }.get(previous_tool, "broad")  # type: ignore[assignment]
                recovery = RetrievalHop(
                    hop_id=f"{hop.hop_id}_alternate_tool",
                    objective=hop.objective,
                    query=str(item.get("executed_query") or hop.query),
                    strategy=recovery_tool,
                    required=False,
                    recovery_for=hop.hop_id,
                )
                plan.hops.append(recovery)
                pending.insert(0, recovery.hop_id)
                self._emit(
                    "recovery_scheduled",
                    hop_id=recovery.hop_id,
                    recovery_for=hop.hop_id,
                    query=recovery.query,
                    strategy=recovery.strategy,
                    recovery_policy="alternate_query_engine",
                )
                return {**state, "plan": plan.model_dump(), "pending_hop_ids": pending}

        sufficient_ids = {hop_id for hop_id, item in ledger.items() if bool(item.get("sufficient"))}
        for hop in plan.hops:
            if hop.recovery_for and hop.hop_id in sufficient_ids:
                sufficient_ids.add(hop.recovery_for)
        all_required_sufficient = all(hop.hop_id in sufficient_ids for hop in required)
        exhausted = len(completed) >= state["max_hops"]
        blocked = bool(state.get("stop_reason"))
        done = all_required_sufficient or exhausted or blocked or not pending
        if not done:
            return {**state, "plan": plan.model_dump(), "pending_hop_ids": pending}

        hop_result_sets = {
            hop_id: [SearchResult.model_validate(item) for item in items]
            for hop_id, items in state.get("hop_results", {}).items()
        }
        final_results = assemble_agent_context(state["query"], hop_result_sets, limit=10)
        stop_reason = (
            "sufficient"
            if all_required_sufficient
            else "hop_budget_exhausted"
            if exhausted
            else state.get("stop_reason") or "subquestions_exhausted"
        )
        duration_ms = round((perf_counter() - float(state["started_at"])) * 1000, 2)
        trace = {
            "policy": "llamaindex_subquestion",
            "mode": plan.mode,
            "rationale": plan.rationale,
            "plan": plan.model_dump(),
            "max_hops": state["max_hops"],
            "completed_hops": completed,
            "pending_hops": pending,
            "evidence_ledger": ledger,
            "sufficient": all_required_sufficient,
            "stop_reason": stop_reason,
            "duration_ms": duration_ms,
            "cost": {
                "retrieval_calls": len(completed),
                "llm_token_estimate": sum(len(str(item.get("executed_query") or "")) for item in ledger.values()) // 4,
            },
        }
        self._emit(
            "retrieval_completed",
            sufficient=all_required_sufficient,
            stop_reason=stop_reason,
            duration_ms=duration_ms,
            result_count=len(final_results),
            results=[result.model_dump() for result in final_results],
            trace=trace,
        )
        return {
            **state,
            "plan": plan.model_dump(),
            "pending_hop_ids": pending,
            "retrieval_results": [result.model_dump() for result in final_results],
            "retrieval_trace": trace,
            "sufficient": all_required_sufficient,
            "stop_reason": stop_reason,
            "duration_ms": duration_ms,
        }

    @staticmethod
    def should_continue(state: AgenticState) -> str:
        return "done" if state.get("retrieval_results") is not None else "continue"


def build_langgraph_agentic_retriever(
    *,
    controller: AgenticRetrievalController | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    use_llm: bool = True,
):
    controller = controller or AgenticRetrievalController(use_llm=use_llm, event_callback=event_callback)
    graph = StateGraph(AgenticState)
    graph.add_node("plan_retrieval", controller.initialize)
    graph.add_node("retrieve_hop", controller.execute_next)
    graph.add_node("judge_sufficiency", controller.judge)
    graph.add_edge(START, "plan_retrieval")
    graph.add_edge("plan_retrieval", "retrieve_hop")
    graph.add_edge("retrieve_hop", "judge_sufficiency")
    graph.add_conditional_edges(
        "judge_sufficiency",
        controller.should_continue,
        {"continue": "retrieve_hop", "done": END},
    )
    return graph.compile()


def _run_coroutine_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Synchronous LlamaIndex retrieval cannot run inside an active event loop.")


def build_llamaindex_agentic_retriever(
    *,
    controller: AgenticRetrievalController | LlamaIndexAgenticController | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    use_llm: bool = True,
):
    try:
        from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
    except ImportError as exc:  # pragma: no cover - exercised in dependency-failure environments
        raise RuntimeError("LlamaIndex agentic retrieval requires llama-index-core.") from exc

    active_controller = controller or LlamaIndexAgenticController(
        use_llm=use_llm,
        event_callback=event_callback,
    )

    class RetrievalEvent(Event):
        state: dict[str, Any]

    class JudgeEvent(Event):
        state: dict[str, Any]

    class ManualsRetrievalWorkflow(Workflow):
        @step
        async def plan(self, event: StartEvent) -> RetrievalEvent:
            state = active_controller.initialize(dict(event.get("state") or {}))
            return RetrievalEvent(state=state)

        @step
        async def retrieve_hop(self, event: RetrievalEvent) -> JudgeEvent:
            return JudgeEvent(state=active_controller.execute_next(event.state))

        @step
        async def judge_sufficiency(self, event: JudgeEvent) -> RetrievalEvent | StopEvent:
            state = active_controller.judge(event.state)
            if active_controller.should_continue(state) == "done":
                return StopEvent(result=state)
            return RetrievalEvent(state=state)

    workflow = ManualsRetrievalWorkflow(timeout=None, verbose=False)

    class SyncLlamaIndexRetriever:
        def invoke(self, state: AgenticState) -> AgenticState:
            async def run_workflow() -> AgenticState:
                return await workflow.run(state=dict(state))

            return _run_coroutine_sync(run_workflow())

    return SyncLlamaIndexRetriever()


def compare_agentic_backends(
    query: str,
    corpus_ids: list[str],
    filters: dict[str, object],
    *,
    max_hops: int = 4,
    use_llm: bool = True,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for backend, factory in (
        ("langgraph", build_langgraph_agentic_retriever),
        ("llamaindex", build_llamaindex_agentic_retriever),
    ):
        started = perf_counter()
        state = factory(use_llm=use_llm).invoke(
            {
                "query": query,
                "corpus_ids": corpus_ids,
                "filters": filters,
                "max_hops": max_hops,
            }
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        outputs[backend] = {
            "elapsed_ms": elapsed_ms,
            "sufficient": state.get("sufficient", False),
            "stop_reason": state.get("stop_reason"),
            "result_chunk_ids": [item["chunk_id"] for item in state.get("retrieval_results", [])],
            "result_document_ids": sorted(
                {item["source_document_id"] for item in state.get("retrieval_results", [])}
            ),
            "results": list(state.get("retrieval_results", [])),
            "trace": state.get("retrieval_trace", {}),
        }
    outputs["equivalent_result_chunks"] = (
        outputs["langgraph"]["result_chunk_ids"] == outputs["llamaindex"]["result_chunk_ids"]
    )
    return outputs
