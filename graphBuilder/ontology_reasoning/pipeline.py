"""High-level OWL-constructed NLP pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import combined_description, decompose_description
from .config import get_profile
from .llm_mapper import map_layers_with_llm  # [UNUSED] build_ontology_context_string
from .retrieval import retrieve_matches
# [UNUSED] from .layer_axioms import is_layer_function_iri
from .material_axioms import resolve_category_and_type
from .validators import (
    apply_named_material_rules,
    expand_allowed_iris,
    extract_thickness_cm,
    validate_composition,
    validate_layer_json,
)


@dataclass
class OntologyNLPContext:
    profile: str
    subject_uri: str
    german_desc: str = ""
    english_desc: str = ""
    chunks: list[str] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    # [UNUSED] assigned once and never read; llm_mapper builds its own prompt block
    # ontology_context: str = ""
    layer_json: dict | list | None = None
    element_thickness_cm: float | None = None
    material_category_iri: str | None = None
    material_type_iri: str | None = None


def _fallback_layer_json(german_desc, english_desc, matches):
    combined = combined_description(german_desc, english_desc)
    if not combined:
        return None

    function_iri = None
    category_iri = None
    for match in matches:
        if match.get("taxonomy_branch") == "Function" and not function_iri:
            function_iri = match["entity_uri"]
        if match.get("taxonomy_branch") == "Material" and not category_iri:
            category_iri = match["entity_uri"]

    if not function_iri and not category_iri:
        return None

    thickness = extract_thickness_cm(combined)
    layer = {
        "layer_index": 1,
        "predicted_function_iri": function_iri,
        "predicted_category_iri": category_iri,
        "predicted_type_iri": None,
        "thickness_cm": thickness,
    }
    return {
        "layer_topology": "https://w3id.org/bmp#SingleLayer",
        "thickness_cm": thickness,
        "layers": [layer],
    }


def process_description(
    *,
    subject_uri: str,
    german_desc: str | None,
    english_desc: str | None,
    ontology_corpus,
    profile: str = "bbsr",
    extra_text: str | None = None,
    use_llm: bool | None = None,
    ontology_graph=None,
    element_type_iri: str | None = None,
):
    """Run steps 2–6 for one description-bearing subject."""
    combined = combined_description(german_desc, english_desc)
    if not combined and not extra_text:
        return None

    cfg = get_profile(profile)
    chunks = decompose_description(german_desc, english_desc, profile=profile, extra_text=extra_text)
    matches = retrieve_matches(
        chunks,
        ontology_corpus,
        profile=profile,
        german_desc=german_desc,
        english_desc=english_desc,
    )

    ctx = OntologyNLPContext(
        profile=profile,
        subject_uri=subject_uri,
        german_desc=german_desc or "",
        english_desc=english_desc or "",
        chunks=chunks,
        matches=matches,
        # [UNUSED] this built the full prompt block, stored it, and discarded it —
        # a second, identical build already happens inside map_layers_with_llm.
        # ontology_context=build_ontology_context_string(
        #     matches,
        #     ontology_graph=ontology_graph,
        #     ontology_corpus=ontology_corpus,
        #     profile=profile,
        # ),
    )

    allowed = expand_allowed_iris(matches, ontology_graph, profile)
    llm_enabled = cfg["use_llm"] if use_llm is None else use_llm

    raw_json = None
    if llm_enabled:
        raw_json = map_layers_with_llm(
            subject_uri=subject_uri,
            german_desc=german_desc or "",
            english_desc=english_desc or "",
            ontology_matches=matches,
            profile=profile,
            ontology_graph=ontology_graph,
            ontology_corpus=ontology_corpus,
            element_type_iri=element_type_iri,
        )

    if cfg["llm_mode"] == "material_only":
        if isinstance(raw_json, dict):
            ctx.material_category_iri = raw_json.get("material_category_iri") or raw_json.get(
                "predicted_category_iri"
            )
            ctx.material_type_iri = raw_json.get("material_type_iri") or raw_json.get("predicted_type_iri")
            # Whitelist: LLM may only return IRIs that retrieval scoped
            if allowed:
                if ctx.material_category_iri and ctx.material_category_iri not in allowed:
                    ctx.material_category_iri = None
                if ctx.material_type_iri and ctx.material_type_iri not in allowed:
                    ctx.material_type_iri = None
        # Fallback to OWL retrieval when LLM is off, failed (e.g. 429), or returned invalid IRIs
        if not ctx.material_category_iri and matches:
            material_matches = [m for m in matches if m.get("taxonomy_branch") == "Material"]
            if material_matches:
                ctx.material_category_iri = material_matches[0]["entity_uri"]
                ctx.material_type_iri = None
        if ontology_graph is not None and (ctx.material_category_iri or ctx.material_type_iri):
            cat_uri, typ_uri = resolve_category_and_type(
                ontology_graph,
                category_iri=ctx.material_category_iri,
                type_iri=ctx.material_type_iri,
            )
            if cat_uri is not None:
                ctx.material_category_iri = str(cat_uri)
                ctx.material_type_iri = str(typ_uri) if typ_uri is not None else None
        return ctx if (ctx.material_category_iri or ctx.material_type_iri) else None

    if cfg["llm_mode"] == "full" and isinstance(raw_json, dict):
        element_thickness, layer_sets = validate_composition(raw_json, allowed, ontology_graph)
        if layer_sets:
            ctx.element_thickness_cm = element_thickness
            ctx.layer_json = _finalize_layer_sets(
                layer_sets,
                german_desc=german_desc,
                english_desc=english_desc,
                ontology_graph=ontology_graph,
                allowed=allowed,
            )
            return ctx if ctx.layer_json else None

    if cfg["llm_mode"] == "full" and isinstance(raw_json, list):
        layer_sets = [
            validate_layer_json(item, allowed, ontology_graph) for item in raw_json
        ]
        ctx.layer_json = _finalize_layer_sets(
            [item for item in layer_sets if item],
            german_desc=german_desc,
            english_desc=english_desc,
            ontology_graph=ontology_graph,
            allowed=allowed,
        )
        return ctx if ctx.layer_json else None

    if isinstance(raw_json, dict):
        ctx.layer_json = validate_layer_json(raw_json, allowed, ontology_graph)
    if not ctx.layer_json:
        ctx.layer_json = validate_layer_json(
            _fallback_layer_json(german_desc, english_desc, matches),
            allowed,
            ontology_graph,
        )
    if isinstance(ctx.layer_json, dict):
        ctx.layer_json = _finalize_layer_sets(
            [ctx.layer_json],
            german_desc=german_desc,
            english_desc=english_desc,
            ontology_graph=ontology_graph,
            allowed=allowed,
        )
    elif isinstance(ctx.layer_json, list):
        ctx.layer_json = _finalize_layer_sets(
            ctx.layer_json,
            german_desc=german_desc,
            english_desc=english_desc,
            ontology_graph=ontology_graph,
            allowed=allowed,
        )

    return ctx if ctx.layer_json else None


def _finalize_layer_sets(layer_sets, *, german_desc, english_desc, ontology_graph, allowed):
    filled = apply_named_material_rules(
        layer_sets,
        german_desc=german_desc,
        english_desc=english_desc,
        ontology_graph=ontology_graph,
    )
    _, validated = validate_composition({"layer_sets": filled}, allowed, ontology_graph)
    return validated


def run_owl_nlp_pipeline(description_record, ontology_corpus, profile: str = "bbsr"):
    """Adapter-friendly wrapper around process_description."""
    return process_description(
        subject_uri=description_record.subject_uri,
        german_desc=description_record.german_desc,
        english_desc=description_record.english_desc,
        ontology_corpus=ontology_corpus,
        profile=profile,
        extra_text=getattr(description_record, "extra_text", None),
        use_llm=getattr(description_record, "use_llm", None),
        element_type_iri=description_record.metadata.get("element_type_iri"),
    )
