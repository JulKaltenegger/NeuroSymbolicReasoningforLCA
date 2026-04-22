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
- `owl/`: ontology folder for LCA. The knowledge base will also be stored here, combining more ontologies and DL rules.
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