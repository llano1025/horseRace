from neo4j import GraphDatabase
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
from typing import List, Dict
import os
import time
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
# # Replace with your Neo4j connection details
# uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "password"
#
# # Create a Neo4j driver instance
# driver = GraphDatabase.driver(uri, auth=(user, password))
#
#
#
#
# def query_to_dataframe(query):
#     with driver.session() as session:
#         # Execute the Cypher query
#         result = session.run(query)
#
#         # Convert the result to a list of dictionaries
#         data = [record.data() for record in result]
#
#         # Convert the list of dictionaries to a DataFrame
#         df = pd.DataFrame(data)
#
#         return df
#
#
# # Query to extract data
# query = """
# MATCH (h:Horse)-[p:PARTICIPATED_IN]->(r:Race)
# MATCH (j:Jockey)-[q:RIDDEN]->(h)
# MATCH (t:Trainer)-[s:TRAINED]->(h)
# MATCH (r:Race)
# WHERE r.date > date("2024-06-01")
# RETURN h, p, r, j, t, q, s, p.placement AS placement, p.horse_weight AS horse_weight, p.draw AS draw
# ORDER BY r.date ASC
# """
#
# # Get the result as a DataFrame
# data = query_to_dataframe(query)
#
# # Create a NetworkX graph from the data
# G = nx.Graph()
#
# # Add nodes and edges to the graph
# for index, row in data.iterrows():
#     G.add_node(row['h']['horse_name'], entity='horse', sex=row['h']['sex'], origin=row['h']['origin'], colour=row['h']['colour'])
#     G.add_node(row['r']['race_name'], entity='race', date=row['r']['date'], location=row['r']['location'], course=row['r']['course'], distance=row['r']['distance'], condition=row['r']['condition'], class_=row['r']['class'])
#     G.add_node(row['j']['jockey_name'], entity='jockey')
#     G.add_node(row['t']['trainer_name'], entity='trainer')
#     G.add_edge(row['p'][0]['horse_name'], row['p'][2]['race_name'], relation='participated', weight=row['horse_weight'], draw=row['draw'])
#     G.add_edge(row['q'][0]['jockey_name'], row['q'][2]['horse_name'], relation='ridden')
#     G.add_edge(row['s'][0]['trainer_name'], row['s'][2]['horse_name'], relation='trained')
#
#
# def normalize_node_attributes(G, required_attrs):
#     for node, attrs in G.nodes(data=True):
#         for attr in required_attrs:
#             if attr not in attrs:
#                 G.nodes[node][attr] = 0
#
#
# def normalize_edge_attributes(G, required_attrs):
#     for u, v, attrs in G.edges(data=True):
#         for attr in required_attrs:
#             if attr not in attrs:
#                 G.edges[u, v][attr] = 0
#
#
# # Convert NetworkX graph to PyTorch Geometric Data object
# node_attrs = ['entity', 'sex', 'origin', 'colour', 'date', 'location', 'course', 'distance', 'condition', 'class_']
# edge_attrs = ['relation', 'weight', 'draw']
# normalize_node_attributes(G, node_attrs)
# normalize_edge_attributes(G, edge_attrs)
#
# # Convert NetworkX graph to PyTorch Geometric Data object
# data = from_networkx(G)

# # Ensure the node feature matrix `x` is set correctly
# node_attrs = ['sex', 'origin', 'colour', 'date', 'location', 'course', 'distance', 'condition', 'class_']
# node_features = []
# for attr in node_attrs:
#     node_features.append([attrs.get(attr, 0) for _, attrs in G.nodes(data=True)])
#
# data.x = torch.tensor(node_features, dtype=torch.float).t()
#
# # Ensure the edge weight matrix `edge_weight` is set correctly if required
# edge_attrs = ['weight', 'draw']
# edge_features = []
# for attr in edge_attrs:
#     edge_features.append([attrs.get(attr, 0) for _, _, attrs in G.edges(data=True)])
#
# data.edge_attr = torch.tensor(edge_features, dtype=torch.float).t()
#
# # Check the attributes
# print(data)


class HorseRacingDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HorseRacingDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['horse_racing_data.pt']

    def process(self):
        # Query to extract data
        query = """
        MATCH (h:Horse)-[p:PARTICIPATED_IN]->(r:Race)
        MATCH (j:Jockey)-[q:RIDDEN]->(h)
        MATCH (t:Trainer)-[s:TRAINED]->(h)
        MATCH (r:Race)
        WHERE r.date > date("2024-06-01")
        RETURN h, p, r, j, t, q, s, p.placement AS placement, p.horse_weight AS horse_weight, p.draw AS draw
        ORDER BY r.date ASC
        """

        # Get the result as a DataFrame
        data = self.query_to_dataframe(query)

        # Create a NetworkX graph from the data
        G = nx.Graph()

        # Add nodes and edges to the graph
        for index, row in data.iterrows():
            G.add_node(row['h']['horse_name'], entity='horse', sex=row['h']['sex'], origin=row['h']['origin'],
                       colour=row['h']['colour'])
            G.add_node(row['r']['race_name'], entity='race', date=row['r']['date'], location=row['r']['location'],
                       course=row['r']['course'], distance=row['r']['distance'], condition=row['r']['condition'],
                       class_=row['r']['class'])
            G.add_node(row['j']['jockey_name'], entity='jockey')
            G.add_node(row['t']['trainer_name'], entity='trainer')
            G.add_edge(row['p'][0]['horse_name'], row['p'][2]['race_name'], relation='participated',
                       weight=row['horse_weight'], draw=row['draw'])
            G.add_edge(row['q'][0]['jockey_name'], row['q'][2]['horse_name'], relation='ridden')
            G.add_edge(row['s'][0]['trainer_name'], row['s'][2]['horse_name'], relation='trained')

        self.normalize_node_attributes(G,
                                       ['entity', 'sex', 'origin', 'colour', 'date', 'location', 'course', 'distance',
                                        'condition', 'class_'])
        self.normalize_edge_attributes(G, ['relation', 'weight', 'draw'])

        # Convert NetworkX graph to PyTorch Geometric Data object
        data = from_networkx(G)

        # If there's only one graph, we need to make it a list
        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def normalize_node_attributes(self, G: nx.Graph, required_attrs: List[str]):
        for node, attrs in G.nodes(data=True):
            for attr in required_attrs:
                if attr not in attrs:
                    G.nodes[node][attr] = 0

    def normalize_edge_attributes(self, G: nx.Graph, required_attrs: List[str]):
        for u, v, attrs in G.edges(data=True):
            for attr in required_attrs:
                if attr not in attrs:
                    G.edges[u, v][attr] = 0

    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        # Your Neo4j connection details
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"

        # Create a Neo4j driver instance
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # Execute the Cypher query
            result = session.run(query)
            data = [record.data() for record in result]
            df = pd.DataFrame(data)

            return df


# Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, '..', 'data/pyG_process/')
dataset = HorseRacingDataset(root=path)
data = dataset[0]

# Prepare features and labels
num_features = data.num_node_features
num_classes = len(torch.unique(data.y))

# Split the data into train, validation, and test sets
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[:int(0.6 * num_nodes)] = True
val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
test_mask[int(0.8 * num_nodes):] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

train_dataset = data.train_mask
val_dataset = data.val_mask
test_dataset = data.test_mask

# Group all training graphs into a single graph to perform sampling:
train_data = Batch.from_data_list(train_dataset)
loader = LinkNeighborLoader(train_data, batch_size=2048, shuffle=True,
                            neg_sampling_ratio=1.0, num_neighbors=[10, 10],
                            num_workers=6, persistent_workers=True)

# Evaluation loaders (one datapoint corresponds to a graph)
train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=64,
    num_layers=2,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = total_examples = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        h = model(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()

    return total_loss / total_examples


@torch.no_grad()
def encode(loader):
    model.eval()

    xs, ys = [], []
    for data in loader:
        data = data.to(device)
        xs.append(model(data.x, data.edge_index).cpu())
        ys.append(data.y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def test():
    # Train classifier on training set:
    x, y = encode(train_loader)

    clf = MultiOutputClassifier(SGDClassifier(loss='log_loss', penalty='l2'))
    clf.fit(x, y)

    train_f1 = f1_score(y, clf.predict(x), average='micro')

    # Evaluate on validation set:
    x, y = encode(val_loader)
    val_f1 = f1_score(y, clf.predict(x), average='micro')

    # Evaluate on test set:
    x, y = encode(test_loader)
    test_f1 = f1_score(y, clf.predict(x), average='micro')

    return train_f1, val_f1, test_f1


times = []
for epoch in range(1, 6):
    start = time.time()
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    train_f1, val_f1, test_f1 = test()
    print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
          f'Test F1: {test_f1:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")