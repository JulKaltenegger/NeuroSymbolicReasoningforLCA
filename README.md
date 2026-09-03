# NeuroSymbolicReasoningforLCA

This project tests **graph-based reasoning methods** for Life Cycle Assessment (LCA).

## Methods

### a) Symbolic-driven
Symbolic rules dominate; neural models fill gaps.  
Best for very low data (our case).

### b) Neural-driven
Neural predictions dominate; symbolic checks are secondary.  
Needs more data.

### c) Hybrid (iterative)
Neural and symbolic components interact continuously.  
Most powerful, but also most complex.

| Aspect | Symbolic | Neural | Neuro-Symbolic |
| --- | --- | --- | --- |
| Representation | RDF + ontology | embeddings / vectors | both |
| Reasoning type | logical inference | statistical learning | hybrid |
| Data requirement | very low | high | low |
| Explainability | high | low | medium |
| Robustness to missing data | low | medium | high |
| Consistency guarantees | strict | none | enforced via rules |
| Suitability for your case | strong baseline | weak alone | strongest |

## Data Requirements

- `data-bim/`: BIM data source.
- `data-lca/`: LCA data source.
- `owl/`: ontology folder for LCA. **Main knowledge base:** `owl/KB-LCA-merged.ttl` (loaded by default). Optional local overlay via `ONTOLOGY_OVERLAY_PATH` in `.env`.
- `ttl/`: RDF data graphs, embedded graph vectors, results, and related outputs.

## Setup

1. Create a virtual environment:  
   `python -m venv .venv`
2. Activate the virtual environment (PowerShell):  
   `.venv\Scripts\Activate.ps1`
3. Install required packages:  
   `pip install -r requirements.txt`

## Goal

Provide a simple environment to test and compare symbolic, neural, and neuro-symbolic graph reasoning methods for LCA.


# File Structure

### graphBuilder (description → TTL)

Entry point: `python graphBuilder/run_pipeline.py`

| Path | Role |
| --- | --- |
| `run_pipeline.py` | Orchestrator: gold text examples, then BBSR / TABULA / SLiCE |
| `sources/run_text_examples.py` | Gold free-text cases → `ttl/text_examples/` |
| `sources/Graph_builder_BBSR_JSON.py` | BBSR JSON → `ttl/bbsr_buildings-enriched.ttl` |
| `sources/Graph_builder_TABULA.py` | TABULA JSON → `ttl/tabula_buildings-enriched.ttl` |
| `sources/Graph_builder_SLICE_CSV.py` | SLiCE CSV → `ttl/slice_data_instantiated.ttl` |
| `adapters/` | Source-specific description records for the shared NLP pipeline |
| `ontology_reasoning/` | Shared OWL + LLM mapping used by every source |
| `owl/shapes/composition.shacl.ttl` | Structural SHACL for LayerSet / Layer / Material |

    python graphBuilder/run_pipeline.py                 # gold examples only
    python graphBuilder/run_pipeline.py --all           # gold + BBSR + TABULA + SLiCE
    python graphBuilder/run_pipeline.py --check-llm     # .env / API key check
    python graphBuilder/run_pipeline.py --check-llm --live
    python graphBuilder/run_pipeline.py --validate      # SHACL + OWL ranges on existing TTL

### ontology_reasoning
| File | Role |
| --- | --- |
| config.py | Profiles (bbsr/tabula/slice), `.env`, LLM provider, `DEFAULT_ONTOLOGY_PATH` → `owl/KB-LCA-merged.ttl` |
| corpus.py | Load OWL graph + build embedded ontology_corpus |
| chunking.py | Text → retrieval chunks |
| retrieval.py | Embedding search → scoped IRI matches |
| element_hints.py | You edit — per beo: element hints for LLM |
| llm_mapper.py | System prompt + user prompt + map_layers_with_llm() |
| llm_backends.py | OpenAI / Google / Ollama JSON chat |
| check_llm.py | `.env` / API key check (`run_pipeline.py --check-llm`) |
| check_ttl.py | OWL ranges + SHACL (`run_pipeline.py --validate`) |
| validators.py | JSON cleanup, OWL whitelist, composition validation |
| ontology_utils.py | Shared OWL helpers (`is_subclass_of`) |
| layer_axioms.py | LayerSet topologies + LayerFunction IRIs and resolution (from OWL) |
| material_axioms.py | Material category/type IRIs and resolution (from OWL) |
| owl_schema.py | TTL skeleton, OWL axiom summaries, scoped vocabulary for LLM prompts |
| rdf_layers.py | emit_enforced_layer(), material instance triples |
| pipeline.py | process_description() — main entry |