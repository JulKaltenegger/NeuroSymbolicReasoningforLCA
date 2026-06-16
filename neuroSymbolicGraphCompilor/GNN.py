import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# Attempt to load torch_geometric dependencies
try:
    import torch_geometric.transforms as T
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError:
    raise ImportError("Please install PyTorch Geometric: `pip install torch-geometric` to run this network.")

BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base Directory: {BASE_DIR}")

# Align paths to your localized workspace hierarchy
SLICE_CSV_PATH = BASE_DIR / "data" / "data_csv_SLiCE" / "SLiCE_Mouton2022a_dropNAN.csv"
TABULA_AB_JSON_PATH = BASE_DIR / "data" / "data-json-TABULA" / "apartment" / "ab_buildings_combined.json"
TABULA_MFH_JSON_PATH = BASE_DIR / "data" / "data-json-TABULA" / "multifamilyhouse" / "mfh_buildings_combined.json"
BBSR_JSON_PATH = BASE_DIR / "data" / "data_text_BBSR" / "json_outputs" / "page_boxes.json"

# ==========================================
# 1. DATA LOADING AND SEMANTIC TOPOLOGY
# ==========================================

def load_all_datasets():
    """Loads CSV and JSON datasets containing structural and LCA definitions."""
    slice_df = pd.read_csv(SLICE_CSV_PATH)
    # Remove nulls before constructing graph layers
    slice_df = slice_df.dropna(subset=["element_name", "techflow_name", "techflow_amount", "indicator_GWP", "techflow_unit"])
    
    with open(TABULA_AB_JSON_PATH, "r", encoding="utf-8") as f:
        ab_data = json.load(f)
    with open(TABULA_MFH_JSON_PATH, "r", encoding="utf-8") as f:
        mfh_data = json.load(f)
    with open(BBSR_JSON_PATH, "r", encoding="utf-8") as f:
        bbsr_data = json.load(f)
        
    return slice_df, ab_data, mfh_data, bbsr_data


def generate_hetero_graph():
    """Parses construction assets into a semantically wired Heterogeneous Graph."""
    slice_df, ab_data, mfh_data, bbsr_data = load_all_datasets()
    data = HeteroData()
    
    # Strip any trailing whitespace from unit categories
    slice_df['techflow_unit'] = slice_df['techflow_unit'].str.strip()
    
    # Setup nominal string encoders
    le_b_type = LabelEncoder().fit(['apartment_building', 'multifamily_house', 'bbsr_typology'])
    le_e_type = LabelEncoder().fit(slice_df['element_name'].unique())
    le_mat_type = LabelEncoder().fit(slice_df['techflow_name'].unique())
    le_unit = LabelEncoder().fit(slice_df['techflow_unit'].unique())
    
    # --- Node Initialization ---
    # 1. Building Nodes: [Building Type Class, Scaled Floor Area]
    buildings_list = ab_data.get('buildings', []) + mfh_data.get('buildings', [])
    b_features = []
    
    for b in buildings_list:
        b_info = b.get('building', {})
        b_type_idx = 0 if b_info.get('size_class') == 'AB' else 1
        area = float(b_info.get('reference_floor_area_m2', 2000.0))
        b_features.append([b_type_idx, area])
    
    for page in bbsr_data.get('pages', []):
        b_features.append([2, 3000.0]) # Typology fallback assignments
        
    b_features = np.array(b_features, dtype=np.float32)
    b_features[:, 1] = StandardScaler().fit_transform(b_features[:, 1].reshape(-1, 1)).flatten()
    data['building'].x = torch.tensor(b_features, dtype=torch.float)
    
    # 2. Element Nodes: [Element Type ID, Service Lifetime Value]
    unique_elements = slice_df['element_name'].unique()
    el_to_idx = {name: i for i, name in enumerate(unique_elements)}
    e_features = [[le_e_type.transform([el])[0], 50.0] for el in unique_elements]
    data['element'].x = torch.tensor(e_features, dtype=torch.float)
    
    # 3. Material Node Mappings: [Material Type ID, Unit ID, Scaled Basic Quantity]
    m_features = []
    targets_gwp = []
    e_to_m_edges = []
    
    # Target transformation (Crucial Assumption Fix 1)
    df_targets_log = np.log10(slice_df['indicator_GWP'].values)
    amounts_scaled = StandardScaler().fit_transform(slice_df['techflow_amount'].values.reshape(-1, 1)).flatten()
    
    for mat_idx, (_, row) in enumerate(slice_df.iterrows()):
        mat_encoded = le_mat_type.transform([row['techflow_name']])[0]
        unit_encoded = le_unit.transform([row['techflow_unit']])[0]
        
        m_features.append([mat_encoded, unit_encoded, amounts_scaled[mat_idx]])
        targets_gwp.append(df_targets_log[mat_idx])
        
        # Build Element -> Material composition topology
        e_idx = el_to_idx[row['element_name']]
        e_to_m_edges.append([e_idx, mat_idx])
        
    data['material'].x = torch.tensor(m_features, dtype=torch.float)
    data['material'].y = torch.tensor(targets_gwp, dtype=torch.float).unsqueeze(1)
    
    # --- Semantic Edge Topology Generation (Crucial Assumption Fix 2) ---
    b_to_e_edges = []
    for b_idx, b in enumerate(buildings_list):
        b_elements = b.get('elements', {})
        for b_el_key in b_elements.keys():
            key_lower = b_el_key.lower()
            for e_name, e_idx in el_to_idx.items():
                if 'wall' in key_lower and ('EW' in e_name or 'IW' in e_name):
                    b_to_e_edges.append([b_idx, e_idx])
                elif ('floor' in key_lower or 'roof' in key_lower) and ('IF' in e_name or 'FR' in e_name):
                    b_to_e_edges.append([b_idx, e_idx])
                    
    if len(b_to_e_edges) == 0:  # Logical fallback strategy to protect graph connectivity
        b_to_e_edges = [[i % data['building'].x.size(0), j] for j in range(len(unique_elements)) for i in range(2)]
                
    data['building', 'contains', 'element'].edge_index = torch.tensor(b_to_e_edges, dtype=torch.long).t().contiguous()
    data['element', 'composed_of', 'material'].edge_index = torch.tensor(e_to_m_edges, dtype=torch.long).t().contiguous()
    
    return data, len(le_mat_type.classes_), len(le_unit.classes_), len(le_e_type.classes_), len(le_b_type.classes_)

# ==========================================
# 2. ENHANCED RESIDUAL HETEROGNN
# ==========================================

class RobustLCAHeteroGNN(nn.Module):
    def __init__(self, hidden_channels, num_m_classes, num_u_classes, num_e_classes, num_b_classes, out_channels=1):
        super().__init__()
        
        # Semantic projection using Embedding layers instead of treat IDs as continuous numbers
        self.b_embed = nn.Embedding(num_b_classes + 1, hidden_channels // 2)
        self.e_embed = nn.Embedding(num_e_classes + 1, hidden_channels // 2)
        self.m_embed = nn.Embedding(num_m_classes + 1, hidden_channels // 2)
        self.u_embed = nn.Embedding(num_u_classes + 1, hidden_channels // 4)
        
        self.b_dense = nn.Linear(1, hidden_channels // 2)
        self.e_dense = nn.Linear(1, hidden_channels // 2)
        self.m_dense = nn.Linear(1, hidden_channels // 4)
        
        # Message passing over heterogeneous boundaries
        self.conv1 = HeteroConv({
            ('building', 'contains', 'element'): SAGEConv((-1, -1), hidden_channels),
            ('element', 'composed_of', 'material'): SAGEConv((-1, -1), hidden_channels)
        })
        
        self.conv2 = HeteroConv({
            ('building', 'contains', 'element'): SAGEConv((-1, -1), hidden_channels),
            ('element', 'composed_of', 'material'): SAGEConv((-1, -1), hidden_channels)
        })
        
        # Non-linear regression head utilizing Skip Connections to retain specific properties
        raw_feature_dim = (hidden_channels // 2) + (hidden_channels // 4) + (hidden_channels // 4)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels + raw_feature_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        # Extract and bundle features
        b_cat = self.b_embed(x_dict['building'][:, 0].long())
        b_num = self.b_dense(x_dict['building'][:, 1].unsqueeze(1))
        h_building = torch.cat([b_cat, b_num], dim=-1)
        
        e_cat = self.e_embed(x_dict['element'][:, 0].long())
        e_num = self.e_dense(x_dict['element'][:, 1].unsqueeze(1))
        h_element = torch.cat([e_cat, e_num], dim=-1)
        
        m_cat = self.m_embed(x_dict['material'][:, 0].long())
        u_cat = self.u_embed(x_dict['material'][:, 1].long())
        m_num = self.m_dense(x_dict['material'][:, 2].unsqueeze(1))
        h_material_raw = torch.cat([m_cat, u_cat, m_num], dim=-1) # Skip Connection source cached
        
        # Instantiate dict mappings for HeteroConv tracking
        mapped_dict = {
            'building': F.relu(h_building),
            'element': F.relu(h_element),
            'material': F.relu(h_material_raw)
        }
        
        # Graph Message Aggregation Rounds
        out_dict = self.conv1(mapped_dict, edge_index_dict)
        mapped_dict.update({key: F.relu(x) for key, x in out_dict.items()})
        
        out_dict = self.conv2(mapped_dict, edge_index_dict)
        mapped_dict.update({key: F.relu(x) for key, x in out_dict.items()})
        
        # Bypassing the over-smoothing trap by concatenating structural data with local properties
        final_material_repr = torch.cat([mapped_dict['material'], h_material_raw], dim=-1)
        
        return self.regression_head(final_material_repr)

# ==========================================
# 3. SPLITTING, TRAINING & METRIC EVALUATION
# ==========================================

def execute_gnn_pipeline():
    graph_data, num_m, num_u, num_e, num_b = generate_hetero_graph()
    
    num_instances = graph_data['material'].x.size(0)
    indices = np.arange(num_instances)
    
    # 70/15/15 Data Mask split boundaries
    train_idx, test_val_idx = train_test_split(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.50, random_state=42)
    
    train_mask = torch.zeros(num_instances, dtype=torch.bool)
    val_mask = torch.zeros(num_instances, dtype=torch.bool)
    test_mask = torch.zeros(num_instances, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    model = RobustLCAHeteroGNN(
        hidden_channels=64, 
        num_m_classes=num_m, 
        num_u_classes=num_u, 
        num_e_classes=num_e, 
        num_b_classes=num_b
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    
    print(f"Dataset compiled.")
    print(f"-> Training Instances:   {train_mask.sum().item()}")
    print(f"-> Validation Instances: {val_mask.sum().item()}")
    print(f"-> Testing Instances:    {test_mask.sum().item()}\n")
    
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(graph_data.x_dict, graph_data.edge_index_dict)
        loss = loss_fn(predictions[train_mask], graph_data['material'].y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(graph_data.x_dict, graph_data.edge_index_dict)[val_mask].cpu().numpy()
                val_targets = graph_data['material'].y[val_mask].cpu().numpy()
                val_mse = mean_squared_error(val_targets, val_preds)
                val_r2 = r2_score(val_targets, val_preds)
            print(f"Epoch {epoch:03d} | Train Log-MSE: {loss.item():.4f} | Val Log-MSE: {val_mse:.4f} | Val Log-R²: {val_r2:.4f}")

    # ==========================================
    # 4. FINAL POST-TRAINING TEST EVALUATION
    # ==========================================
    model.eval()
    with torch.no_grad():
        final_out = model(graph_data.x_dict, graph_data.edge_index_dict)
        
        y_val_true_log = graph_data['material'].y[val_mask].cpu().numpy()
        y_val_pred_log = final_out[val_mask].cpu().numpy()
        
        y_test_true_log = graph_data['material'].y[test_mask].cpu().numpy()
        y_test_pred_log = final_out[test_mask].cpu().numpy()
        
    # Log scale performance
    val_r2_log = r2_score(y_val_true_log, y_val_pred_log)
    test_r2_log = r2_score(y_test_true_log, y_test_pred_log)
    test_mse_log = mean_squared_error(y_test_true_log, y_test_pred_log)
    
    # Back-transform to view real-scale performance
    y_test_true_orig = 10 ** y_test_true_log
    y_test_pred_orig = 10 ** y_test_pred_log
    test_r2_orig = r2_score(y_test_true_orig, y_test_pred_orig)
    test_mse_orig = mean_squared_error(y_test_true_orig, y_test_pred_orig)
    
    print("\n" + "="*50)
    print("         FINAL MODEL PERFORMANCE METRICS        ")
    print("="*50)
    print(f"LOG-SCALE EVALUATION (Order of Magnitude Accuracy):")
    print(f"  - Validation R² Score:                 {val_r2_log:.4f}")
    print(f"  - Unseen Test R² Score:                {test_r2_log:.4f}")
    print(f"  - Unseen Test Mean Squared Error:       {test_mse_log:.4f}")
    print("-" * 50)
    print(f"ORIGINAL UNTRANSFORMED SCALE EVALUATION:")
    print(f"  - Unseen Test R² Score:                {test_r2_orig:.4f}")
    print(f"  - Unseen Test Mean Squared Error:       {test_mse_orig:.4f}")
    print("="*50)

if __name__ == '__main__':
    execute_gnn_pipeline()