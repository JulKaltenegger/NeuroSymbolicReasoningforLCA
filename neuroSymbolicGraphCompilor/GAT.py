import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

try:
    import rdflib
    from rdflib import RDF, RDFS, Namespace, Graph
    import torch_geometric
    import torch_geometric.transforms as T 
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, GATConv
except ImportError:
    raise ImportError("Missing required packages. Please run: pip install rdflib torch-geometric scikit-learn sentence-transformers")

# ==========================================
# FILE PATH & LOCAL COORD SYSTEM STORAGE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"Base Directory: {BASE_DIR}")

TABULA_TTL = BASE_DIR / "ttl" / "tabula_buildings-enriched.ttl"
BBSR_TTL = BASE_DIR / "ttl" / "bbsr_buildings-enriched.ttl"
SLICE_TTL = BASE_DIR / "ttl" / "slice_data_instantiated.ttl"

# ==========================================
# MATHEMATICAL UTILITY REGEX PARSER
# ==========================================
def extract_numeric_literal(val_literal):
    if not val_literal:
        return 1.0
    val_str = str(val_literal).strip()
    try:
        return float(val_str)
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str.replace(',', '.'))
        if nums:
            return np.mean([float(n) for n in nums])
    return 1.0

# =====================================================================
# PHASE 1 ONLY: MULTI-MODAL ONTOLOGY PARSER (FÄRBER ET AL. ARCHITECTURE)
# =====================================================================
def feature_initalisation():
    g = Graph()
    print("Parsing semantic Turtle data layers into active memory...")
    g.parse(str(TABULA_TTL), format="turtle")
    g.parse(str(BBSR_TTL), format="turtle")
    g.parse(str(SLICE_TTL), format="turtle")
    print(f"Knowledge Graph operational with {len(g)} total RDF statements.")
    
    data = HeteroData()
    
    # 1. Comprehensive Namespace Declarations
    BMP = Namespace("https://w3id.org/bmp#")
    AT = Namespace("https://w3id.org/at#")
    BOT = Namespace("https://w3id.org/bot#")
    LCA = Namespace("https://w3id.org/lca#")
    BEO = Namespace("https://w3id.org/beo#")
    BPO = Namespace("https://w3id.org/bpo#")
    SLICE = Namespace("https://w3id.org/slice#")
    TABULA = Namespace("https://namedgraphs.org/tabula#")
    BBSR = Namespace("https://namedgraphs.org/bbsr#")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    QUDT = Namespace("http://qudt.org/schema/qudt/")
    OM = Namespace("http://ontology.eil.utoronto.ca/icity/OM#")
    DC1 = Namespace("http://purl.org/dc/elements/1.1/identifier")
    UNIT = Namespace("http://qudt.org/vocab/unit/")
    RDF_VAL = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#value")
    
    # 2. SEPARATED STREAM CONTAINERS PER ENTITY TYPE
    # xs Track: Structural Tracking Registries (Global Graph Coordinate Identification)
    building_uris, element_uris, layerset_uris, layer_uris, material_uris, stage_uris = [], [], [], [], [], []
    
    # xc Track: Lexical/Text Containers (Raw strings awaiting BERT embedding processing)
    building_xc, element_xc, layerset_xc, material_xc, stage_xc = [], [], [], [], []
    
    # xt Track: Literal Numerical Containers (Continuous features awaiting scaling/normalization)
    building_xt, element_xt, layer_xt, stage_xt = [], [], [], []
    
    # Target Tracker (Isolated ground-truth dependent variable for prediction)
    targets_stage_gwp = []

    print("\n[PHASE 1] Initializing Independent Multi-Modal Streams: xc (Text), xt (Numeric), xs (Structural)")

    # --- 1. Process Building Entities ---
    # Filter predicate: # rdf:type -> bot:Building
    for s in g.subjects(RDF.type, BOT.Building):
        uri = str(s)
        if uri not in building_uris:
            building_uris.append(uri)                                           # -> xs Track (URI String)
            
            # Content Property: # rdfs:label
            label = str(g.value(s, RDFS.label) or "Building Archetype")         # -> xc Track (Raw Text)
            
            # Numerical Property: # at:hasReferenceFloorArea
            area_node = g.value(s, AT.hasReferenceFloorArea)
            area = extract_numeric_literal(g.value(area_node, RDF_VAL) if area_node else g.value(s, AT.hasReferenceFloorArea)) # -> xt Track (Float)
            
            # Numerical Property: # bot:hasStorey
            storey_val = g.value(s, BOT.hasStorey)
            storeys = extract_numeric_literal(storey_val) if storey_val else 3.0 # -> xt Track (Float)
            
            # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
            building_xc.append(label)             
            building_xt.append([area, storeys])   
            
    print(f" -> Buildings tracked independently: {len(building_xc)} text items, {len(building_xt)} numeric rows.")

    # --- 2. Process Element Entities ---
    # Filter predicate: # bmp:hasLayerSet
    for s in g.subjects(BMP.hasLayerSet, None):
        uri = str(s)
        if uri not in element_uris:
            element_uris.append(uri)                                            # -> xs Track (URI String)
            
            # Content Property: # at:hasArchetypeDescription or # rdfs:label
            desc = str(g.value(s, AT.hasArchetypeDescription) or g.value(s, RDFS.label) or "Component Element") # -> xc Track (Raw Text)
            
            # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
            element_xc.append(desc)               
            element_xt.append([50.0])             # -> xt Track (Constant fallback float for design lifespan)

    # --- 3. Process Construction Layer Candidate Entities ---
    # Filter predicate: # rdf:type -> bmp:LayerSet
    for s in g.subjects(RDF.type, BMP.LayerSet):
        uri = str(s)
        if uri not in layerset_uris:
            layerset_uris.append(uri)                                           # -> xs Track (URI String)
            
            # Content Property: # rdfs:label
            desc = str(g.value(s, RDFS.label) or "Assembly Construction Layer") # -> xc Track (Raw Text)
            
            # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
            layerset_xc.append(desc)              

    # --- 4. Process Layer Entities ---
    # Filter predicate: # rdf:type -> bmp:Layer
    for s in g.subjects(RDF.type, BMP.Layer):
        uri = str(s)
        if uri not in layer_uris:
            layer_uris.append(uri)                                              # -> xs Track (URI String)
            
            # Numerical Property: # bmp:hasThickness
            thick_node = g.value(s, BMP.hasThickness)
            thickness = extract_numeric_literal(g.value(thick_node, RDF_VAL) if thick_node else None) # -> xt Track (Float)
            
            # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
            layer_xt.append([thickness])          

    # --- 5. Process Material and LifeCycle Stage Entities ---
    l_to_m_edges = []
    m_to_stage_edges = []
    
    for l_idx, l_uri in enumerate(layer_uris):
        l_node = rdflib.URIRef(l_uri)
        # Structural Linkage Predicate: # bmp:hasMaterial
        for mat_node in g.objects(l_node, BMP.hasMaterial):
            mat_uri = str(mat_node)
            
            if mat_uri not in material_uris:
                material_uris.append(mat_uri)                                   # -> xs Track (URI String)
                
                # Content Property: # bmp:hasMaterialCategory
                cat_uri = g.value(mat_node, BMP.hasMaterialCategory)
                mat_cat = str(cat_uri).split("#")[-1] if cat_uri else "Concrete" # -> xc Track (Raw Text)
                
                # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
                material_xc.append(mat_cat)       

            mat_idx = material_uris.index(mat_uri)
            l_to_m_edges.append([l_idx, mat_idx])                               
            
            # Structural Linkage Predicate: # bmp:hasLifeCycleStage
            for stage_node in g.objects(mat_node, BMP.hasLifeCycleStage):
                stage_uri = str(stage_node)
                
                # Target Regression Numerical Property: # lca:hasGWP
                gwp_literal = g.value(stage_node, LCA.hasGWP)
                
                if gwp_literal:
                    gwp_val = float(gwp_literal)
                    
                    # Content Property 1: # rdf:type (Extracting LCA stage code e.g. A1A3, C4)
                    stage_types = [str(t).split("#")[-1] for t in g.objects(stage_node, RDF.type) if "owl" not in str(t).lower()]
                    stage_type_str = stage_types[0] if stage_types else "A1A3"   # -> xc Track Part 1 (Raw Text)
                    
                    # Content Property 2: # lca:hasActivityType
                    activity_uri = g.value(stage_node, LCA.hasActivityType)
                    activity_str = str(activity_uri).split("#")[-1] if activity_uri else "MaterialIn" # -> xc Track Part 2 (Raw Text)
                    
                    # --- ASYNCHRONOUS STORAGE TRACK ASSIGNMENT (HOP 0) ---
                    stage_xc.append([stage_type_str, activity_str]) 
                    targets_stage_gwp.append(np.log10(max(gwp_val, 1e-6)))       
                    
                    stage_uris.append(stage_uri)
                    stage_idx = len(stage_uris) - 1
                    m_to_stage_edges.append([mat_idx, stage_idx])               

    print(f" -> Materials tracked independently: {len(material_xc)} text descriptors.")
    print(f" -> LifeCycle Stages tracked independently: {len(stage_xc)} text token pairs.")

# --- 6. Topology Connection Extraction (xs structural linkage processing) ---
    b_map = {uri: i for i, uri in enumerate(building_uris)}
    e_map = {uri: i for i, uri in enumerate(element_uris)}
    ls_map = {uri: i for i, uri in enumerate(layerset_uris)}
    l_map = {uri: i for i, uri in enumerate(layer_uris)}
    
    b_to_e, e_to_ls, ls_to_l = [], [], []
    for s, p, o in g.triples((None, None, None)):
        if str(s) in b_map and str(o) in e_map: b_to_e.append([b_map[str(s)], e_map[str(o)]])
    for s, p, o in g.triples((None, BMP.hasLayerSet, None)):
        if str(s) in e_map and str(o) in ls_map: e_to_ls.append([e_map[str(s)], ls_map[str(o)]])
    for s, p, o in g.triples((None, BMP.hasLayer, None)):
        if str(s) in ls_map and str(o) in l_map: ls_to_l.append([ls_map[str(s)], l_map[str(o)]])

    # === PACKAGING AT THE BOTTOM OF PHASE 1 ===
    print("\n[PHASE 1 COMPLETION] Packaging isolated streams into separate Graph attributes...")

    # Storing xc (Lexical Text Content Streams)
    data['building'].xc = building_xc
    data['element'].xc = element_xc
    data['layerset'].xc = layerset_xc
    data['material'].xc = material_xc
    data['stage'].xc = stage_xc

    # Storing xt (Numeric Literal Streams - tensors of floats)
    data['building'].xt = torch.tensor(building_xt, dtype=torch.float)
    data['element'].xt = torch.tensor(element_xt, dtype=torch.float)
    data['layer'].xt = torch.tensor(layer_xt, dtype=torch.float)

    # Storing Ground Truth Target Vector
    data['stage'].y = torch.tensor(targets_stage_gwp, dtype=torch.float).unsqueeze(1)
    
    # Constructing structural topological tracks (xs stream)
    data['building', 'contains', 'element'].edge_index = torch.tensor(b_to_e, dtype=torch.long).t().contiguous()
    data['element', 'utilizes', 'layerset'].edge_index = torch.tensor(e_to_ls, dtype=torch.long).t().contiguous()
    data['layerset', 'composed_of', 'layer'].edge_index = torch.tensor(ls_to_l, dtype=torch.long).t().contiguous()
    data['layer', 'contains_material', 'material'].edge_index = torch.tensor(l_to_m_edges, dtype=torch.long).t().contiguous()
    data['material', 'has_stage', 'stage'].edge_index = torch.tensor(m_to_stage_edges, dtype=torch.long).t().contiguous()
    
    # Decoder A classification ground truth
    lset_ground_truth = np.zeros(len(element_uris), dtype=np.int64)
    for src, dst in e_to_ls: lset_ground_truth[src] = dst
    data['element'].y = torch.tensor(lset_ground_truth, dtype=torch.long)

    # xs URI registries per node type (input to TransE in Phase 2)
    data["building"].uris = building_uris
    data["element"].uris = element_uris
    data["layerset"].uris = layerset_uris
    data["layer"].uris = layer_uris
    data["material"].uris = material_uris
    data["stage"].uris = stage_uris

    # Global KG entity index for TransE: h_head + r ≈ h_tail
    all_uris = building_uris + element_uris + layerset_uris + layer_uris + material_uris + stage_uris
    uri_to_global = {uri: idx for idx, uri in enumerate(all_uris)}
    rel_to_id = {}
    transe_triples = []
    for s, p, o in g.triples((None, None, None)):
        s_uri, o_uri = str(s), str(o)
        if s_uri not in uri_to_global or o_uri not in uri_to_global:
            continue
        p_uri = str(p)
        if p_uri not in rel_to_id:
            rel_to_id[p_uri] = len(rel_to_id)
        transe_triples.append((uri_to_global[s_uri], rel_to_id[p_uri], uri_to_global[o_uri]))

    data.num_kg_entities = len(all_uris)
    data.num_kg_relations = max(len(rel_to_id), 1)
    data.transe_triples = transe_triples
    data.uri_to_global = uri_to_global
    print(f" -> TransE KG: {data.num_kg_entities} entities, {len(rel_to_id)} relations, {len(transe_triples)} triples.")

    data = T.ToUndirected()(data)
    
    print("PyTorch Geometric LifeCycle Assessment Graph compiled successfully.")
    return data, len(layerset_uris)

# ==========================================
# PHASE 2: DIMENSION SYNCHRONIZATION (Hop-0 → d)
#   2a  xs  TransE URI geometry     →  Z_s
#   2b  xc  SentenceTransformer     →  Z_c   (text only — not numerics)
#   2c  xt  StandardScaler + W_τ    →  Z_t
#   edge_index is NOT Phase 2: it is the graph skeleton consumed in Phase 3 (GAT).
#   τ = RDF node class (building, element, …); W_τ and b_τ are per-class Linear weights.
# ==========================================
TRANSE_DIM = 32
TRANSE_EPOCHS = 80
TRANSE_MARGIN = 1.0


class TransE(nn.Module):
    """Structural xs encoder: geometric coordinates for URI entities."""

    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, heads, relations, tails):
        return (self.entity(heads) + self.relation(relations) - self.entity(tails)).norm(p=2, dim=1)


def _pretrain_transe(triples, num_entities, num_relations, dim=TRANSE_DIM, epochs=TRANSE_EPOCHS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransE(num_entities, num_relations, dim).to(device)
    if len(triples) < 1:
        return model.cpu()

    heads = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
    rels = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
    tails = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"[PHASE 2a] Pre-training TransE ({epochs} epochs, dim={dim})...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pos = model.score(heads, rels, tails)
        neg_tails = torch.randint(0, num_entities, tails.shape, device=device)
        neg = model.score(heads, rels, neg_tails)
        loss = F.relu(pos - neg + TRANSE_MARGIN).mean()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"  TransE epoch {epoch:03d} | margin loss {loss.item():.4f}")

    return model.cpu()


def _assign_transe_node_embeddings(graph_data, transe_model):
    uri_to_global = graph_data.uri_to_global
    dim = transe_model.entity.embedding_dim
    for ntype in ("building", "element", "layerset", "layer", "material", "stage"):
        uris = getattr(graph_data[ntype], "uris", [])
        if not uris:
            graph_data[ntype].xs_emb = torch.zeros(0, dim)
            continue
        ids = torch.tensor([uri_to_global[u] for u in uris], dtype=torch.long)
        graph_data[ntype].xs_emb = transe_model.entity(ids).detach()


def _encode_texts(texts, model, device):
    if not texts:
        dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
        return torch.zeros(0, dim)
    return model.encode(texts, convert_to_tensor=True, device=device).cpu()


def dimension_synchronization(graph_data):
    """Fuse xc, xt, xs into per-node tensors ready for Phase 3 GAT (hidden dim d)."""
    from sentence_transformers import SentenceTransformer

    # --- Phase 2a: xs via TransE (structural semantic coordinates) ---
    transe_model = _pretrain_transe(
        graph_data.transe_triples,
        graph_data.num_kg_entities,
        graph_data.num_kg_relations,
    )
    _assign_transe_node_embeddings(graph_data, transe_model)
    transe_dim = transe_model.entity.embedding_dim

    # --- Phase 2b: xc via language model (strings only) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[PHASE 2b] Encoding xc with {DEFAULT_EMBEDDER} on {device}...")
    embedder = SentenceTransformer(DEFAULT_EMBEDDER, device=device)
    embed_dim = embedder.get_embedding_dimension() if hasattr(embedder, "get_embedding_dimension") else embedder.get_sentence_embedding_dimension()

    graph_data["building"].text_emb = _encode_texts(graph_data["building"].xc, embedder, device)
    graph_data["element"].text_emb = _encode_texts(graph_data["element"].xc, embedder, device)
    graph_data["layerset"].text_emb = _encode_texts(graph_data["layerset"].xc, embedder, device)
    graph_data["material"].text_emb = _encode_texts(graph_data["material"].xc, embedder, device)

    stage_types = [pair[0] for pair in graph_data["stage"].xc]
    stage_activities = [pair[1] for pair in graph_data["stage"].xc]
    type_vocab = {label: idx for idx, label in enumerate(sorted(set(stage_types)))}
    act_vocab = {label: idx for idx, label in enumerate(sorted(set(stage_activities)))}
    graph_data["stage"].type_id = torch.tensor([type_vocab[s] for s in stage_types], dtype=torch.long)
    graph_data["stage"].act_id = torch.tensor([act_vocab[s] for s in stage_activities], dtype=torch.long)

    # --- Phase 2c: xt via scaling (numerics never pass through BERT) ---
    print("[PHASE 2c] Scaling xt numeric streams...")
    for node_type in ("building", "element", "layer"):
        xt = graph_data[node_type].xt.numpy()
        if len(xt):
            graph_data[node_type].xt = torch.tensor(
                StandardScaler().fit_transform(xt), dtype=torch.float
            )

    print("[PHASE 2] Synchronized stream shapes (xc / xt / xs):")
    for node_type in ("building", "element", "layerset", "layer", "material", "stage"):
        if node_type == "stage":
            print(f"  -> {node_type}: xs {tuple(graph_data[node_type].xs_emb.shape)}, "
                  f"xc categoricals ({len(type_vocab)} types, {len(act_vocab)} activities)")
            continue
        xs = graph_data[node_type].xs_emb
        print(f"  -> {node_type}: xs {tuple(xs.shape)}", end="")
        if hasattr(graph_data[node_type], "text_emb"):
            print(f" | xc {tuple(graph_data[node_type].text_emb.shape)}", end="")
        if hasattr(graph_data[node_type], "xt"):
            print(f" | xt {tuple(graph_data[node_type].xt.shape)}", end="")
        print()

    return graph_data, embed_dim, transe_dim, len(type_vocab), len(act_vocab)


# Backward-compatible alias
neural_encoding = dimension_synchronization


def _link_topk_accuracy(score_matrix, element_indices, true_layerset_ids, k=2):
    """Top-k link prediction: rank all LayerSets for each element."""
    if len(element_indices) == 0:
        return 0.0, 0.0
    scores = score_matrix[element_indices]
    true = true_layerset_ids
    top1 = (scores.argmax(dim=1) == true).float().mean().item() * 100
    k = min(k, scores.size(1))
    top2_hits = sum(1 for i, t in enumerate(true) if t in scores[i].topk(k).indices)
    return top1, (top2_hits / len(true)) * 100


# ==========================================
# PHASE 3–4: MULTI-TASK HETEROGENEOUS GAT
# ==========================================
class ElementLayerSetLinkDecoder(nn.Module):
    """Decoder A: bilinear link predictor score(h_element, h_layerset)."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_channels, hidden_channels, 1)

    def score_all_pairs(self, h_element, h_layerset):
        n_e, n_ls = h_element.size(0), h_layerset.size(0)
        h_e = h_element.unsqueeze(1).expand(n_e, n_ls, -1)
        h_ls = h_layerset.unsqueeze(0).expand(n_e, n_ls, -1)
        return self.bilinear(h_e, h_ls).squeeze(-1)

    def score_edges(self, h_element, h_layerset, edge_index):
        src, dst = edge_index
        return self.bilinear(h_element[src], h_layerset[dst]).squeeze(-1)


class MultiTaskHeteroGAT(nn.Module):
    def __init__(self, hidden_channels, embed_dim, transe_dim, num_stage_types,
                 num_activity_types, edge_types, heads=2):
        super().__init__()
        third = hidden_channels // 3
        two_thirds = hidden_channels - third

        # Per-class W_τ projections: Z_v = concat(Z_c, Z_t, Z_s) → R^d
        self.b_text_proj = nn.Linear(embed_dim, third)
        self.b_num_proj = nn.Linear(2, third)
        self.b_struct_proj = nn.Linear(transe_dim, hidden_channels - 2 * third)

        self.e_text_proj = nn.Linear(embed_dim, third)
        self.e_num_proj = nn.Linear(1, third)
        self.e_struct_proj = nn.Linear(transe_dim, hidden_channels - 2 * third)

        self.ls_text_proj = nn.Linear(embed_dim, two_thirds)
        self.ls_struct_proj = nn.Linear(transe_dim, hidden_channels - two_thirds)

        self.l_num_proj = nn.Linear(1, two_thirds)
        self.l_struct_proj = nn.Linear(transe_dim, hidden_channels - two_thirds)

        self.mat_text_proj = nn.Linear(embed_dim, two_thirds)
        self.mat_struct_proj = nn.Linear(transe_dim, hidden_channels - two_thirds)

        self.stg_type_embed = nn.Embedding(max(num_stage_types, 1), third)
        self.stg_act_embed = nn.Embedding(max(num_activity_types, 1), third)
        self.stg_struct_proj = nn.Linear(transe_dim, hidden_channels - 2 * third)

        self.gat1 = HeteroConv({
            et: GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
            for et in edge_types
        })
        self.gat2 = HeteroConv({
            et: GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
            for et in edge_types
        })
        self.dim_reduction = nn.Linear(hidden_channels * heads, hidden_channels)

        self.decoder_a_link = ElementLayerSetLinkDecoder(hidden_channels)
        self.decoder_b_regression = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def _hop0_project(self, data):
        h_b = F.relu(torch.cat([
            self.b_text_proj(data["building"].text_emb),
            self.b_num_proj(data["building"].xt),
            self.b_struct_proj(data["building"].xs_emb),
        ], dim=-1))
        h_e = F.relu(torch.cat([
            self.e_text_proj(data["element"].text_emb),
            self.e_num_proj(data["element"].xt),
            self.e_struct_proj(data["element"].xs_emb),
        ], dim=-1))
        h_ls = F.relu(torch.cat([
            self.ls_text_proj(data["layerset"].text_emb),
            self.ls_struct_proj(data["layerset"].xs_emb),
        ], dim=-1))
        h_l = F.relu(torch.cat([
            self.l_num_proj(data["layer"].xt),
            self.l_struct_proj(data["layer"].xs_emb),
        ], dim=-1))
        h_mat = F.relu(torch.cat([
            self.mat_text_proj(data["material"].text_emb),
            self.mat_struct_proj(data["material"].xs_emb),
        ], dim=-1))
        h_stg_raw = torch.cat([
            self.stg_type_embed(data["stage"].type_id),
            self.stg_act_embed(data["stage"].act_id),
            self.stg_struct_proj(data["stage"].xs_emb),
        ], dim=-1)
        h_stg_raw = F.relu(h_stg_raw)
        return {
            "building": h_b, "element": h_e, "layerset": h_ls,
            "layer": h_l, "material": h_mat, "stage": h_stg_raw,
        }, h_stg_raw

    def encode(self, data):
        latent_dict, h_stg_raw = self._hop0_project(data)
        out1 = self.gat1(latent_dict, data.edge_index_dict)
        latent_dict = {k: F.relu(self.dim_reduction(v)) for k, v in out1.items() if k in latent_dict}
        out2 = self.gat2(latent_dict, data.edge_index_dict)
        latent_dict = {k: F.relu(self.dim_reduction(v)) for k, v in out2.items() if k in latent_dict}
        return latent_dict, h_stg_raw

    def forward(self, data):
        latent_dict, h_stg_raw = self.encode(data)
        link_scores = self.decoder_a_link.score_all_pairs(
            latent_dict["element"], latent_dict["layerset"]
        )
        fused_stage = torch.cat([latent_dict["stage"], h_stg_raw], dim=-1)
        predicted_gwp = self.decoder_b_regression(fused_stage)
        return link_scores, predicted_gwp


def _link_prediction_loss(link_decoder, h_element, h_layerset, edge_index, train_src_mask, num_layersets):
    """Margin-based link loss on element→layerset positive edges with tail corruption."""
    src, dst = edge_index
    train_edge_mask = train_src_mask[src]
    if train_edge_mask.sum() == 0:
        return torch.tensor(0.0, device=h_element.device)

    pos_src, pos_dst = src[train_edge_mask], dst[train_edge_mask]
    pos_scores = link_decoder.score_edges(h_element, h_layerset, torch.stack([pos_src, pos_dst]))
    neg_dst = torch.randint(0, num_layersets, pos_dst.shape, device=h_element.device)
    neg_scores = link_decoder.score_edges(
        h_element, h_layerset, torch.stack([pos_src, neg_dst])
    )
    return F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()


# Alias for notebooks / earlier references
MaterialLCACAHeteroGAT = MultiTaskHeteroGAT


# ==========================================
# 3. PIPELINE OPTIMIZATION ENGINE
# ==========================================
def run_gat_pipeline():
    graph_data, _num_lset_classes = feature_initalisation()
    graph_data, embed_dim, transe_dim, num_stage_types, num_activity_types = dimension_synchronization(graph_data)

    num_elements = len(graph_data["element"].xc)
    num_stages = len(graph_data["stage"].xc)
    num_layersets = len(graph_data["layerset"].xc)

    print(f"\n[PIPELINE] Preparing splits for {num_elements} Elements and {num_stages} LifeCycle Stages.")
    e_train_idx, e_temp_idx = train_test_split(np.arange(num_elements), test_size=0.30, random_state=42)
    e_val_idx, e_test_idx = train_test_split(e_temp_idx, test_size=0.50, random_state=42)
    stg_train_idx, stg_temp_idx = train_test_split(np.arange(num_stages), test_size=0.30, random_state=42)
    stg_val_idx, stg_test_idx = train_test_split(stg_temp_idx, test_size=0.50, random_state=42)

    e_train_mask = torch.zeros(num_elements, dtype=torch.bool)
    e_train_mask[e_train_idx] = True

    print(f"-> Elements: train={len(e_train_idx)}, val={len(e_val_idx)}, test={len(e_test_idx)}")
    print(f"-> Stages:   train={len(stg_train_idx)}, val={len(stg_val_idx)}, test={len(stg_test_idx)}")

    model = MultiTaskHeteroGAT(
        hidden_channels=64,
        embed_dim=embed_dim,
        transe_dim=transe_dim,
        num_stage_types=num_stage_types,
        num_activity_types=num_activity_types,
        edge_types=graph_data.edge_types,
        heads=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    loss_regression = nn.SmoothL1Loss()
    e_ls_edge = graph_data["element", "utilizes", "layerset"].edge_index

    print("\n[PHASE 3] Training multi-task heterogeneous GAT (link prediction + GWP regression)...")
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()

        latent_dict, h_stg_raw = model.encode(graph_data)
        loss_a = _link_prediction_loss(
            model.decoder_a_link,
            latent_dict["element"],
            latent_dict["layerset"],
            e_ls_edge,
            e_train_mask,
            num_layersets,
        )
        predicted_gwp = model.decoder_b_regression(
            torch.cat([latent_dict["stage"], h_stg_raw], dim=-1)
        )
        loss_b = loss_regression(predicted_gwp[stg_train_idx], graph_data["stage"].y[stg_train_idx])
        total_loss = loss_a + loss_b
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                link_scores, val_gwp = model(graph_data)
                val_top1, val_top2 = _link_topk_accuracy(
                    link_scores, torch.tensor(e_val_idx), graph_data["element"].y[e_val_idx], k=2
                )
                val_stage_pred = val_gwp[stg_val_idx].cpu().numpy().flatten()
                val_stage_true = graph_data["stage"].y[stg_val_idx].cpu().numpy().flatten()
                val_r2 = r2_score(val_stage_true, val_stage_pred)
                val_mse = mean_squared_error(val_stage_true, val_stage_pred)
            model.train()
            print(
                f"  Epoch {epoch:03d} | Train loss {total_loss.item():.4f} "
                f"(link {loss_a.item():.4f}, GWP {loss_b.item():.4f}) | "
                f"Val link top-1 {val_top1:.2f}% | top-2 {val_top2:.2f}% | "
                f"Val GWP R² {val_r2:.4f} | Val GWP MSE {val_mse:.4f}"
            )

    model.eval()
    with torch.no_grad():
        link_scores, final_stage_preds = model(graph_data)

        val_top1, val_top2 = _link_topk_accuracy(
            link_scores, torch.tensor(e_val_idx), graph_data["element"].y[e_val_idx], k=2
        )
        test_top1, test_top2 = _link_topk_accuracy(
            link_scores, torch.tensor(e_test_idx), graph_data["element"].y[e_test_idx], k=2
        )

        val_pred_stage_log = final_stage_preds[stg_val_idx].cpu().numpy().flatten()
        val_true_stage_log = graph_data["stage"].y[stg_val_idx].cpu().numpy().flatten()
        val_r2_log = r2_score(val_true_stage_log, val_pred_stage_log)
        val_mse_log = mean_squared_error(val_true_stage_log, val_pred_stage_log)

        test_pred_stage_log = final_stage_preds[stg_test_idx].cpu().numpy().flatten()
        test_true_stage_log = graph_data["stage"].y[stg_test_idx].cpu().numpy().flatten()
        test_r2_log = r2_score(test_true_stage_log, test_pred_stage_log)
        test_mse_log = mean_squared_error(test_true_stage_log, test_pred_stage_log)

    print("\n" + "=" * 50)
    print("   MULTI-TASK HETEROGENEOUS GAT METRICS")
    print("=" * 50)
    print("DECODER A: ELEMENT -> LAYERSET LINK PREDICTION (bilinear, top-k)")
    print(f"  - Validation Top-1:  {val_top1:.2f}%")
    print(f"  - Validation Top-2:  {val_top2:.2f}%")
    print(f"  - Test Top-1:          {test_top1:.2f}%")
    print(f"  - Test Top-2:          {test_top2:.2f}%")
    print("-" * 50)
    print("DECODER B: LIFECYCLE STAGE GWP REGRESSION (log10 scale)")
    print(f"  - Validation R²:  {val_r2_log:.4f}")
    print(f"  - Validation MSE: {val_mse_log:.4f}")
    print(f"  - Test R²:        {test_r2_log:.4f}")
    print(f"  - Test MSE:       {test_mse_log:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    run_gat_pipeline()