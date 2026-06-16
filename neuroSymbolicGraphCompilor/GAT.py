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
    raise ImportError("Missing required packages. Please run: pip install rdflib torch-geometric scikit-learn")

# Establish exact localized paths to match your environment layout
BASE_DIR = Path(r"C:\Users\20180031\Dropbox\_PhD\11_KnowledgeBaseLCA\NeuroSymbolicReasoningforLCA")
print(f"Base Directory: {BASE_DIR}")

TABULA_TTL = BASE_DIR / "ttl" / "tabula_buildings-enriched.ttl"
BBSR_TTL = BASE_DIR / "ttl" / "bbsr_buildings-enriched.ttl"
SLICE_TTL = BASE_DIR / "ttl" / "slice_data_instantiated.ttl"

# ==========================================
# HELPER PARSING FUNCTIONS
# ==========================================

def extract_numeric_literal(val_literal):
    """Safely extracts and parses numeric metrics from strings or varying formats."""
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

# ==========================================
# 1. EXPANDED MULTI-MODAL ONTOLOGY PARSER
# ==========================================

def load_and_compile_passport_dataset():
    g = Graph()
    print("Parsing semantic Turtle data layers into active memory...")
    g.parse(str(TABULA_TTL), format="turtle")
    g.parse(str(BBSR_TTL), format="turtle")
    g.parse(str(SLICE_TTL), format="turtle")
    print(f"Knowledge Graph operational with {len(g)} total RDF statements.")
    
    data = HeteroData()
    
    BMP = Namespace("https://w3id.org/bmp#")
    AT = Namespace("https://w3id.org/at#")
    BOT = Namespace("https://w3id.org/bot#")
    LCA = Namespace("https://w3id.org/lca#")
    RDF_VAL = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#value")
    
    building_uris, element_uris, layerset_uris, layer_uris, stage_uris = [], [], [], [] ,[]
    building_features, element_features, layerset_features, layer_features, stage_features = [], [], [], [], []
    targets_stage_gwp = []
    
    # Cache lists to uniquely identify categorical stage strings
    lc_stages_vocab, activity_types_vocab = [], []

    # --- 1. Process Building Entities ---
    for s in g.subjects(RDF.type, BOT.Building):
        uri = str(s)
        if uri not in building_uris:
            building_uris.append(uri)
            label = str(g.value(s, RDFS.label) or "Building Archetype")
            area_node = g.value(s, AT.hasReferenceFloorArea)
            area = extract_numeric_literal(g.value(area_node, RDF_VAL) if area_node else g.value(s, AT.hasReferenceFloorArea))
            storey_val = g.value(s, BOT.hasStorey)
            storeys = extract_numeric_literal(storey_val) if storey_val else 3.0
            building_features.append([hash(label) % 1000, area, storeys])
            
    # --- 2. Process Element Entities ---
    for s in g.subjects(BMP.hasLayerSet, None):
        uri = str(s)
        if uri not in element_uris:
            element_uris.append(uri)
            desc = str(g.value(s, AT.hasArchetypeDescription) or g.value(s, RDFS.label) or "Component Element")
            element_features.append([hash(desc) % 1000, 50.0])
            
    # --- 3. Process LayerSet Candidate Entities ---
    for s in g.subjects(RDF.type, BMP.LayerSet):
        uri = str(s)
        if uri not in layerset_uris:
            layerset_uris.append(uri)
            desc = str(g.value(s, RDFS.label) or "Assembly LayerSet")
            layerset_features.append([hash(desc) % 1000])
            
    # --- 4. Process Layer & Granular Stage Instance Nodes ---
    l_to_stage_edges = []
    for s in g.subjects(RDF.type, BMP.Layer):
        uri = str(s)
        if uri not in layer_uris:
            layer_uris.append(uri)
            thick_node = g.value(s, BMP.hasThickness)
            thickness = extract_numeric_literal(g.value(thick_node, RDF_VAL) if thick_node else None)
            
            mat_node = g.value(s, BMP.hasMaterial)
            mat_cat = "Concrete"
            if mat_node:
                cat_uri = g.value(mat_node, BMP.hasMaterialCategory)
                if cat_uri:
                    mat_cat = str(cat_uri).split("#")[-1]
            
            layer_features.append([hash(mat_cat) % 1000, thickness])
            layer_idx = len(layer_uris) - 1
            
            # Extract Granular Lifecycle Stage Sub-Entities belonging to this layer
            if mat_node:
                for stage_node in g.objects(mat_node, BMP.hasLifeCycleStage):
                    stage_uri = str(stage_node)
                    gwp_literal = g.value(stage_node, LCA.hasGWP)
                    
                    if gwp_literal:
                        gwp_val = float(gwp_literal)
                        # Identify specific lifecycle codes (e.g., A1, A2, A3, C4)
                        stage_types = [str(t).split("#")[-1] for t in g.objects(stage_node, RDF.type) if "owl" not in str(t).lower()]
                        stage_type_str = stage_types[0] if stage_types else "A1A3"
                        
                        activity_uri = g.value(stage_node, LCA.hasActivityType)
                        activity_str = str(activity_uri).split("#")[-1] if activity_uri else "MaterialIn"
                        
                        # Populate vocab arrays dynamically
                        if stage_type_str not in lc_stages_vocab: lc_stages_vocab.append(stage_type_str)
                        if activity_str not in activity_types_vocab: activity_types_vocab.append(activity_str)
                        
                        stage_type_id = lc_stages_vocab.index(stage_type_str)
                        activity_id = activity_types_vocab.index(activity_str)
                        
                        stage_uris.append(stage_uri)
                        stage_features.append([stage_type_id, activity_id])
                        targets_stage_gwp.append(np.log10(max(gwp_val, 1e-6)))
                        
                        # Append structural connection from Layer index to Stage index
                        stage_idx = len(stage_uris) - 1
                        l_to_stage_edges.append([layer_idx, stage_idx])

    # Registry Lookups
    b_map = {uri: i for i, uri in enumerate(building_uris)}
    e_map = {uri: i for i, uri in enumerate(element_uris)}
    ls_map = {uri: i for i, uri in enumerate(layerset_uris)}
    l_map = {uri: i for i, uri in enumerate(layer_uris)}
    
    # --- 5. Extract Topology Connections ---
    b_to_e, e_to_ls, ls_to_l = [], [], []
    for s, p, o in g.triples((None, None, None)):
        if str(s) in b_map and str(o) in e_map:
            b_to_e.append([b_map[str(s)], e_map[str(o)]])
            
    for s, p, o in g.triples((None, BMP.hasLayerSet, None)):
        if str(s) in e_map and str(o) in ls_map:
            e_to_ls.append([e_map[str(s)], ls_map[str(o)]])
            
    for s, p, o in g.triples((None, BMP.hasLayer, None)):
        if str(s) in ls_map and str(o) in l_map:
            ls_to_l.append([ls_map[str(s)], l_map[str(o)]])

    # Convert to standard PyG tensors
    data['building'].x = torch.tensor(building_features, dtype=torch.float)
    data['element'].x = torch.tensor(element_features, dtype=torch.float)
    data['layerset'].x = torch.tensor(layerset_features, dtype=torch.float)
    data['layer'].x = torch.tensor(layer_features, dtype=torch.float)
    
    # Define features and outputs for the newly introduced node type
    data['stage'].x = torch.tensor(stage_features, dtype=torch.float)
    data['stage'].y = torch.tensor(targets_stage_gwp, dtype=torch.float).unsqueeze(1)
    
    data['building', 'contains', 'element'].edge_index = torch.tensor(b_to_e, dtype=torch.long).t().contiguous()
    data['element', 'utilizes', 'layerset'].edge_index = torch.tensor(e_to_ls, dtype=torch.long).t().contiguous()
    data['layerset', 'composed_of', 'layer'].edge_index = torch.tensor(ls_to_l, dtype=torch.long).t().contiguous()
    data['layer', 'has_stage', 'stage'].edge_index = torch.tensor(l_to_stage_edges, dtype=torch.long).t().contiguous()
    
    lset_ground_truth = np.zeros(len(element_uris), dtype=np.int64)
    for src, dst in e_to_ls:
        lset_ground_truth[src] = dst
    data['element'].y = torch.tensor(lset_ground_truth, dtype=torch.long)
    
    # Generate bidirectional messaging tracks across the extended graph
    data = T.ToUndirected()(data)
    
    print("\nPyTorch Geometric Passport Graph compiled successfully:")
    print(f"  -> Nodes: {data['building'].num_nodes} Buildings, {data['element'].num_nodes} Elements, {data['layerset'].num_nodes} LayerSets, {data['layer'].num_nodes} Layers, {data['stage'].num_nodes} MaterialPassportStages.")
    return data, len(layerset_uris)

# ==========================================
# 2. GRANULAR MATERIAL PASSPORT HETEROGAT
# ==========================================

class MaterialPassportLCAHeteroGAT(nn.Module):
    def __init__(self, hidden_channels, num_layerset_classes, edge_types, heads=2):
        super().__init__()
        
        # Linear projection embedding alignment layers
        self.b_embed = nn.Embedding(1005, hidden_channels // 2)
        self.b_dense = nn.Linear(2, hidden_channels // 2)
        
        self.e_embed = nn.Embedding(1005, hidden_channels // 2)
        self.e_dense = nn.Linear(1, hidden_channels // 2)
        
        self.ls_embed = nn.Embedding(1005, hidden_channels)
        
        self.l_embed = nn.Embedding(1005, hidden_channels // 2)
        self.l_dense = nn.Linear(1, hidden_channels // 2)
        
        # Categorical embedding alignment layers for the new Stage node properties
        self.stg_type_embed = nn.Embedding(100, hidden_channels // 2)
        self.stg_act_embed = nn.Embedding(100, hidden_channels // 2)
        
        # Multi-Head Relation-Aware Graph Attention Convolutions
        self.gat1 = HeteroConv({et: GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False) for et in edge_types})
        self.gat2 = HeteroConv({et: GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False) for et in edge_types})
        
        self.dim_reduction = nn.Linear(hidden_channels * heads, hidden_channels)
        
        # DECODER A: LayerSet Inference Configuration Head
        self.decoder_a_layerset = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_layerset_classes)
        )
        
        # DECODER B: Granular continuous regression head with skip connection
        self.decoder_b_regression = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        # 1. Coordinate alignment projection
        h_b = torch.cat([self.b_embed(x_dict['building'][:, 0].long()), self.b_dense(x_dict['building'][:, 1:3])], dim=-1)
        h_e = torch.cat([self.e_embed(x_dict['element'][:, 0].long()), self.e_dense(x_dict['element'][:, 1].unsqueeze(1))], dim=-1)
        h_ls = self.ls_embed(x_dict['layerset'][:, 0].long())
        h_l_raw = torch.cat([self.l_embed(x_dict['layer'][:, 0].long()), self.l_dense(x_dict['layer'][:, 1].unsqueeze(1))], dim=-1)
        
        # Construct the raw properties vector for the stage node
        h_stg_raw = torch.cat([self.stg_type_embed(x_dict['stage'][:, 0].long()), self.stg_act_embed(x_dict['stage'][:, 1].long())], dim=-1)
        
        latent_dict = {
            'building': F.relu(h_b), 'element': F.relu(h_e), 'layerset': F.relu(h_ls), 'layer': F.relu(h_l_raw), 'stage': F.relu(h_stg_raw)
        }
        
        # 2. Graph Attention messaging loops (Hop 1 & 2)
        out1 = self.gat1(latent_dict, edge_index_dict)
        latent_dict = {k: F.relu(self.dim_reduction(v)) for k, v in out1.items() if k in latent_dict}
        
        out2 = self.gat2(latent_dict, edge_index_dict)
        latent_dict = {k: F.relu(self.dim_reduction(v)) for k, v in out2.items() if k in latent_dict}
        
        # 3. Process Decoders
        pred_layersets = self.decoder_a_layerset(latent_dict['element'])
        
        # Concat multi-hop graph representations with local properties to predict the GWP
        fused_stage_context = torch.cat([latent_dict['stage'], h_stg_raw], dim=-1)
        predicted_gwp = self.decoder_b_regression(fused_stage_context)
        
        return pred_layersets, predicted_gwp

# ==========================================
# 3. UNIFIED PASSPORT OPTIMIZATION ENGINE
# ==========================================

def run_passport_pipeline():
    graph_data, num_lset_classes = load_and_compile_passport_dataset()
    
    num_elements = graph_data['element'].x.size(0)
    num_stages = graph_data['stage'].x.size(0)
    
    e_train_idx, e_test_idx = train_test_split(np.arange(num_elements), test_size=0.2, random_state=42)
    stg_train_idx, stg_test_idx = train_test_split(np.arange(num_stages), test_size=0.2, random_state=42)
    
    model = MaterialPassportLCAHeteroGAT(hidden_channels=64, num_layerset_classes=num_lset_classes, edge_types=graph_data.edge_types, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    
    loss_classification = nn.CrossEntropyLoss()
    loss_regression = nn.SmoothL1Loss()
    
    print("\nStarting Unified Training Loop across Expanded Material Passport Topology...")
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        
        lset_logits, stage_predictions = model(graph_data.x_dict, graph_data.edge_index_dict)
        
        loss_a = loss_classification(lset_logits[e_train_idx], graph_data['element'].y[e_train_idx])
        loss_b = loss_regression(stage_predictions[stg_train_idx], graph_data['stage'].y[stg_train_idx])
        
        total_loss = loss_a + loss_b
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | Layout Loss: {loss_a.item():.4f} | Passport Stage GWP Loss: {loss_b.item():.4f}")
            
    # Final Model Evaluation
    model.eval()
    with torch.no_grad():
        final_lset_logits, final_stage_preds = model(graph_data.x_dict, graph_data.edge_index_dict)
        
        # Evaluate Decoder A
        test_logits = final_lset_logits[e_test_idx].cpu().numpy()
        true_lset = graph_data['element'].y[e_test_idx].cpu().numpy()
        top2_hits = 0
        for i, true_val in enumerate(true_lset):
            if true_val in np.argsort(test_logits[i])[-2:]:
                top2_hits += 1
        top2_accuracy = (top2_hits / len(true_lset)) * 100 if len(true_lset) > 0 else 100.0
        
        # Evaluate Decoder B
        pred_stage_log = final_stage_preds[stg_test_idx].cpu().numpy().flatten()
        true_stage_log = graph_data['stage'].y[stg_test_idx].cpu().numpy().flatten()
        r2_log = r2_score(true_stage_log, pred_stage_log)
        mse_log = mean_squared_error(true_stage_log, pred_stage_log)
        
    print("\n" + "="*50)
    print("      MULTI-TASK MATERIAL PASSPORT METRICS     ")
    print("="*50)
    print(f"DECODER A: ASSEMBLY LAYOUT RECOMMENDATION")
    print(f"  - Top-2 LayerSet Selection Accuracy:     {top2_accuracy:.2f}%")
    print("-" * 50)
    print(f"DECODER B: MATERIAL PASSPORT STAGE REGRESSION")
    print(f"  - Log-Scale R² Score (Order of Mag):    {r2_log:.4f}")
    print(f"  - Log-Scale Mean Squared Error (MSE):    {mse_log:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_passport_pipeline()
    
    
    # import os
# import re
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
# from pathlib import Path

# # ==========================================
# # DEPENDENCY SANITY CHECK & INSTALATION GUIDE
# # ==========================================
# try:
#     import rdflib
#     from rdflib import RDF, RDFS, Namespace, Graph
#     import torch_geometric
#     from torch_geometric.data import HeteroData
#     from torch_geometric.nn import HeteroConv, GATConv
# except ImportError:
#     print("Missing critical Graph or Semantic framework dependencies.")
#     print("Please run the following command in your terminal/environment:")
#     print("  pip install rdflib torch-geometric torch-scatter torch-sparse scikit-learn")
#     raise ImportError("Frameworks not initialized.")


# BASE_DIR = Path(__file__).resolve().parent.parent
# print(f"Base Directory: {BASE_DIR}")

# # Align paths to your localized workspace hierarchy
# BBSR_TTL = BASE_DIR / "ttl" / "bbsr_buildings-enriched.ttl"
# TABULA_TTL = BASE_DIR / "ttl" / "tabula_buildings-enriched.ttl"
# SLICE_TTL = BASE_DIR / "ttl" / "slice_data_instantiated.ttl"

# # ==========================================
# # 1. DATA LOADING AND SEMANTIC TOPOLOGY GRAPH COMPILER
# # ==========================================

# def compile_rdf_knowledge_graph():
#     """Loads and merges your three specialized Turtle files into a unified triple store."""
#     g = Graph()
    
#     # Define filenames exactly as uploaded in the repository directory
#     ttl_files = [TABULA_TTL, BBSR_TTL, SLICE_TTL]
    
#     print("Parsing enriched semantic Turtle files into active memory...")
#     for file in ttl_files:
#         if os.path.exists(file):
#             g.parse(file, format="turtle")
#             print(f"  -> Successfully parsed: {file}")
#         else:
#             print(f"  [Warning] File not located in workspace directory: {file}")
            
#     print(f"Graph fully operational. Loaded {len(g)} explicit RDF statements.\n")
#     return g


# def extract_multi_modal_features_to_pyg():
#     """
#     Parses Datatype and Object properties from RDF to build a PyG HeteroData object.
#     Implements Phase 1 (Feature Initialization) and Phase 2 (Dimension Sync).
#     """
#     g = compile_rdf_knowledge_graph()
#     data = HeteroData()
    
#     # Define Namespace mappings found inside your Turtle prefixes
#     BMP = Namespace("https://w3id.org/bmp#")
#     AT = Namespace("https://w3id.org/at#")
#     BOT = Namespace("https://w3id.org/bot#")
#     LCA = Namespace("https://w3id.org/lca#")
#     RDF_VAL = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#value")

#     # Entity Tracking Cache Arrays
#     building_uris, element_uris, layerset_uris, layer_uris = [], [], [], []
    
#     # ---------------------------------------------------------
#     # EXTRACT ENTITIES & DATATYPE PROPERTIES (Content & Literal Streams)
#     # ---------------------------------------------------------
#     # 1. Building Entities: [BERT Text Content Hash, Tabular Year, Tabular Storeys]
#     b_raw_feats = []
#     for s in g.subjects(RDF.type, BOT.Building):
#         uri = str(s)
#         building_uris.append(uri)
#         label = str(g.value(s, RDFS.label) or "Residential Structure")
#         # Continuous datatype properties parsed directly
#         year = 1978.0 if "06" in label or "70" in label else 1990.0  # Archetype default
#         storeys = 5.0 if "MFH" in label else 2.0
#         b_raw_feats.append([hash(label) % 1000, year, storeys])
        
#     # 2. BuiltElement Entities: [Element Category Semantic Hash, Planned Lifetime]
#     e_raw_feats = []
#     for s in g.subjects(RDF.type, AT.ElementArchetype):
#         uri = str(s)
#         element_uris.append(uri)
#         desc = str(g.value(s, AT.hasArchetypeDescription) or "Component Layout")
#         e_raw_feats.append([hash(desc) % 1000, 50.0]) # 50 Year Service life standard
        
#     # 3. LayerSet Assemblies (Targets for Link Prediction / Decoder A)
#     for s in g.subjects(RDF.type, BMP.LayerSet):
#         layerset_uris.append(str(s))
#     # Fill defaults if LayerSet nodes are standalone identifiers
#     lset_raw_feats = [[i % 10, 1.0] for i in range(len(layerset_uris))]
        
#     # 4. Layer/Material Instances: [Material Sub-Category Hash, Scaled Thickness]
#     l_raw_feats = []
#     targets_gwp = []
#     for s in g.subjects(RDF.type, BMP.Layer):
#         uri = str(s)
#         layer_uris.append(uri)
        
#         # Pull Continuous Property (Thickness Datatype)
#         thick_node = g.value(s, BMP.hasThickness)
#         thickness = float(g.value(thick_node, RDF_VAL) or 10.0) if thick_node else 10.0
        
#         # Pull Decomposed NLP Text Category Description
#         mat_cat = str(g.value(s, BMP.hasMaterialCategory) or "Concrete")
        
#         # Pull target variable for regression (LCA footprint)
#         gwp_val = float(g.value(s, LCA.indicator_GWP) or 1.5)
        
#         l_raw_feats.append([hash(mat_cat) % 1000, thickness])
#         targets_gwp.append(np.log10(max(gwp_val, 1e-6))) # Stabilize target range

#     # Prevent crash on empty mock executions by enforcing baseline shapes
#     if not building_uris: building_uris, b_raw_feats = ["b0"], [[0, 1978, 5]]
#     if not element_uris: element_uris, e_raw_feats = ["e0"], [[1, 50]]
#     if not layerset_uris: layerset_uris, lset_raw_feats = ["ls0", "ls1"], [[0, 1], [1, 1]]
#     if not layer_uris: layer_uris, l_raw_feats, targets_gwp = ["l0"], [[2, 14]], [0.5]

#     # Map URIs to uniform numerical index positions
#     b_map = {uri: i for i, uri in enumerate(building_uris)}
#     e_map = {uri: i for i, uri in enumerate(element_uris)}
#     ls_map = {uri: i for i, uri in enumerate(layerset_uris)}
#     l_map = {uri: i for i, uri in enumerate(layer_uris)}
    
#     # ---------------------------------------------------------
#     # EXTRACT OBJECT PROPERTIES (Relational Topology Tracks)
#     # ---------------------------------------------------------
#     b_to_e, e_to_ls, ls_to_l = [], [], []
    
#     # Trace Building -> BuiltElement Archetypes
#     for s, p, o in g.triples((None, AT.hasElementArchetype, None)):
#         if str(s) in b_map and str(o) in e_map:
#             b_to_e.append([b_map[str(s)], e_map[str(o)]])
            
#     # Trace BuiltElement -> LayerSet Configuration Maps
#     for s, p, o in g.triples((None, BMP.hasLayerSet, None)):
#         if str(s) in e_map and str(o) in ls_map:
#             e_to_ls.append([e_map[str(s)], ls_map[str(o)]])
            
#     # Trace LayerSet -> Material Layer Instances
#     for s, p, o in g.triples((None, BMP.hasLayer, None)):
#         if str(s) in ls_map and str(o) in l_map:
#             ls_to_l.append([ls_map[str(s)], l_map[str(o)]])
            
#     # Structural Check: Ensure connectivity tracks are populated (apply semantic lookups if sparse)
#     if not b_to_e: b_to_e = [[0, 0]]
#     if not e_to_ls: e_to_ls = [[0, 0], [0, 1 % len(layerset_uris)]]
#     if not ls_to_l: ls_to_l = [[i % len(layerset_uris), i % len(layer_uris)] for i in range(len(layer_uris))]

#     # Convert all compiled matrices to PyTorch tensors
#     data['building'].x = torch.tensor(b_raw_feats, dtype=torch.float)
#     data['element'].x = torch.tensor(e_raw_feats, dtype=torch.float)
#     data['layerset'].x = torch.tensor(lset_raw_feats, dtype=torch.float)
#     data['layer'].x = torch.tensor(l_raw_feats, dtype=torch.float)
#     data['layer'].y = torch.tensor(targets_gwp, dtype=torch.float).unsqueeze(1)
    
#     # Establish structural edge arrays
#     data['building', 'contains', 'element'].edge_index = torch.tensor(b_to_e, dtype=torch.long).t().contiguous()
#     data['element', 'utilizes', 'layerset'].edge_index = torch.tensor(e_to_ls, dtype=torch.long).t().contiguous()
#     data['layerset', 'composed_of', 'layer'].edge_index = torch.tensor(ls_to_l, dtype=torch.long).t().contiguous()
    
#     # Assign ground truth labels for Decoder A (LayerSet type classifications)
#     # Each element points to its primary matching structural layout class ID
#     lset_ground_truth = np.zeros(len(element_uris), dtype=np.int64)
#     for src, dst in e_to_ls:
#         lset_ground_truth[src] = dst
#     data['element'].y = torch.tensor(lset_ground_truth, dtype=torch.long)
    
#     print("PyTorch Geometric Dataset compilation completed:")
#     print(f"  Nodes: {len(building_uris)} Buildings, {len(element_uris)} Elements, {len(layerset_uris)} LayerSets, {len(layer_uris)} Layers.")
#     return data, len(layerset_uris)

# # ==========================================
# # 2. MULTI-TASK HETEROGENEOUS GRAPH ATTENTION MODEL
# # ==========================================

# class MultiTaskArchitectureLCAHeteroGAT(nn.Module):
#     def __init__(self, hidden_channels, num_layerset_classes, heads=2):
#         super().__init__()
        
#         # Phase 2: Separate Linear Projection Layers (Dimension Synchronization)
#         # Converted Text Hashes / Categoricals map to dedicated lookup matrices
#         self.b_embed = nn.Embedding(1005, hidden_channels // 2)
#         self.e_embed = nn.Embedding(1005, hidden_channels // 2)
#         self.l_embed = nn.Embedding(1005, hidden_channels // 2)
        
#         self.b_dense = nn.Linear(2, hidden_channels // 2)
#         self.e_dense = nn.Linear(1, hidden_channels // 2)
#         self.lset_proj = nn.Linear(2, hidden_channels)
#         self.l_dense = nn.Linear(1, hidden_channels // 2)
        
#         # Phase 3: Heterogeneous Graph Attention Convolution layers
#         self.gat1 = HeteroConv({
#             ('building', 'contains', 'element'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
#             ('element', 'utilizes', 'layerset'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
#             ('layerset', 'composed_of', 'layer'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
#         })
        
#         self.gat2 = HeteroConv({
#             ('building', 'contains', 'element'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
#             ('element', 'utilizes', 'layerset'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False),
#             ('layerset', 'composed_of', 'layer'): GATConv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
#         })
        
#         self.dim_reduction = nn.Linear(hidden_channels * heads, hidden_channels)
        
#         # Phase 4: Multi-Task Decoder Heads
#         # DECODER A: Softmax Class Prediction (Predict Top 2 LayerSets on Element scale)
#         self.decoder_a_layerset = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, num_layerset_classes)
#         )
        
#         # DECODER B: Continuous Value Regression (Conditioned LCA Carbon Footprint on Material scale)
#         self.decoder_b_lca = nn.Sequential(
#             nn.Linear(hidden_channels + hidden_channels, hidden_channels),
#             nn.LayerNorm(hidden_channels),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=0.1),
#             nn.Linear(hidden_channels, 1)
#         )

#     def forward(self, x_dict, edge_index_dict):
#         # Unpack, transform and project features into a synchronized dimension
#         h_b = torch.cat([self.b_embed(x_dict['building'][:, 0].long()), self.b_dense(x_dict['building'][:, 1:3])], dim=-1)
#         h_e = torch.cat([self.e_embed(x_dict['element'][:, 0].long()), self.e_dense(x_dict['element'][:, 1].unsqueeze(1))], dim=-1)
#         h_ls = self.lset_proj(x_dict['layerset'])
#         h_l_raw = torch.cat([self.l_embed(x_dict['layer'][:, 0].long()), self.l_dense(x_dict['layer'][:, 1].unsqueeze(1))], dim=-1)
        
#         current_space = {
#             'building': F.relu(h_b),
#             'element': F.relu(h_e),
#             'layerset': F.relu(h_ls),
#             'layer': F.relu(h_l_raw)
#         }
        
#         # Multi-Hop Edge Message Passing (Hop 1)
#         out_dict = self.gat1(current_space, edge_index_dict)
#         current_space.update({k: F.relu(self.dim_reduction(v)) for k, v in out_dict.items() if k in current_space})
        
#         # Multi-Hop Edge Message Passing (Hop 2)
#         out_dict = self.gat2(current_space, edge_index_dict)
#         current_space.update({k: F.relu(self.dim_reduction(v)) for k, v in out_dict.items() if k in current_space})
        
#         # Execute Decoders
#         pred_layersets = self.decoder_a_layerset(current_space['element'])
        
#         # Skip Connection: Concatenate multi-hop graph representations with local layer features
#         fused_layer_context = torch.cat([current_space['layer'], h_l_raw], dim=-1)
#         pred_lca_values = self.decoder_b_lca(fused_layer_context)
        
#         return pred_layersets, pred_lca_values

# # ==========================================
# # 3. TRAINING ROUTINE, TESTING & VALIDATION
# # ==========================================

# def run_multitask_lca_pipeline():
#     # 1. Generate standard PyG graph components
#     graph_data, num_lset_classes = extract_multi_modal_features_to_pyg()
    
#     # 2. Establish Mask Splitting across both targets
#     num_elements = graph_data['element'].x.size(0)
#     num_layers = graph_data['layer'].x.size(0)
    
#     e_train_idx, e_test_idx = train_test_split(np.arange(num_elements), test_size=0.2, random_state=42) if num_elements > 1 else ([0], [0])
#     l_train_idx, l_test_idx = train_test_split(np.arange(num_layers), test_size=0.2, random_state=42) if num_layers > 1 else ([0], [0])
    
#     # 3. Initialize Model and Optimizers
#     model = MultiTaskArchitectureLCAHeteroGAT(hidden_channels=64, num_layerset_classes=num_lset_classes)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-3)
    
#     loss_classification = nn.CrossEntropyLoss()
#     loss_regression = nn.SmoothL1Loss() # Huber loss for robust LCA metrics
    
#     print("\nStarting Unified Training Loop...")
#     for epoch in range(1, 121):
#         model.train()
#         optimizer.zero_grad()
        
#         # Execute forward pass across the multi-task pipeline
#         lset_logits, lca_predictions = model(graph_data.x_dict, graph_data.edge_index_dict)
        
#         # Compute multi-task losses simultaneously
#         loss_a = loss_classification(lset_logits[e_train_idx], graph_data['element'].y[e_train_idx])
#         loss_b = loss_regression(lca_predictions[l_train_idx], graph_data['layer'].y[l_train_idx])
        
#         # Combined Backpropagation
#         total_loss = loss_a + loss_b
#         total_loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         if epoch % 20 == 0:
#             print(f"  Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | LayerSet Loss: {loss_a.item():.4f} | GWP Loss: {loss_b.item():.4f}")
            
#     # ---------------------------------------------------------
#     # FINAL MODEL TEST PERFORMANCE EVALUATION
#     # ---------------------------------------------------------
#     model.eval()
#     with torch.no_grad():
#         final_lset_logits, final_lca_preds = model(graph_data.x_dict, graph_data.edge_index_dict)
        
#         # Evaluate Decoder A (Top 2 LayerSet Classifications)
#         test_logits = final_lset_logits[e_test_idx].cpu().numpy()
#         true_lset = graph_data['element'].y[e_test_idx].cpu().numpy()
        
#         # Calculate Top-2 accuracy (is true layerset in the top 2 predicted slots?)
#         top2_hits = 0
#         for i, true_val in enumerate(true_lset):
#             top_2_preds = np.argsort(test_logits[i])[-2:] # Extract top 2 indices
#             if true_val in top_2_preds:
#                 top2_hits += 1
#         top2_accuracy = top2_hits / len(true_lset) if len(true_lset) > 0 else 1.0
        
#         # Evaluate Decoder B (LCA Material Core Footprint)
#         test_lca_preds_log = final_lca_preds[l_test_idx].cpu().numpy().flatten()
#         true_lca_log = graph_data['layer'].y[l_test_idx].cpu().numpy().flatten()
        
#         r2_log = r2_score(true_lca_log, test_lca_preds_log) if len(true_lca_log) > 1 else 1.0
#         mse_log = mean_squared_error(true_lca_log, test_lca_preds_log)
        
#     print("\n" + "="*50)
#     print("      MULTI-TASK HETEROGAT EVALUATION METRICS     ")
#     print("="*50)
#     print(f"DECODER A: ASSEMBLY LAYOUT RECOMMENDATION")
#     print(f"  - Top-2 LayerSet Selection Accuracy:     {top2_accuracy * 100:.2f}%")
#     print("-" * 50)
#     print(f"DECODER B: CONTEXT-AWARE LCA REGRESSION")
#     print(f"  - Log-Scale R² Score (Order of Mag):    {r2_log:.4f}")
#     print(f"  - Log-Scale Mean Squared Error (MSE):    {mse_log:.4f}")
#     print("="*50)

# if __name__ == "__main__":
#     run_multitask_lca_pipeline()