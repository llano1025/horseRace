import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from neo4j import GraphDatabase
import pandas as pd

# Replace with your Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))


def query_to_dataframe(query):
    with driver.session() as session:
        # Execute the Cypher query
        result = session.run(query)

        # Convert the result to a list of dictionaries
        data = [record.data() for record in result]

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        return df


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
data = query_to_dataframe(query)

# Create a NetworkX graph from the data
G = nx.Graph()

# Add nodes and edges to the graph
for index, row in data.iterrows():
    G.add_node(row['h']['horse_name'], entity='horse', sex=row['h']['sex'], origin=row['h']['origin'], colour=row['h']['colour'])
    G.add_node(row['r']['race_name'], entity='race', date=row['r']['date'], location=row['r']['location'], course=row['r']['course'], distance=row['r']['distance'], condition=row['r']['condition'], class_=row['r']['class'])
    G.add_node(row['j']['jockey_name'], entity='jockey')
    G.add_node(row['t']['trainer_name'], entity='trainer')
    G.add_edge(row['p'][0]['horse_name'], row['p'][2]['race_name'], relation='participated', weight=row['horse_weight'], draw=row['draw'])
    G.add_edge(row['q'][0]['jockey_name'], row['q'][2]['horse_name'], relation='ridden')
    G.add_edge(row['s'][0]['trainer_name'], row['s'][2]['horse_name'], relation='trained')


def normalize_node_attributes(G, required_attrs):
    for node, attrs in G.nodes(data=True):
        for attr in required_attrs:
            if attr not in attrs:
                G.nodes[node][attr] = 0


def normalize_edge_attributes(G, required_attrs):
    for u, v, attrs in G.edges(data=True):
        for attr in required_attrs:
            if attr not in attrs:
                G.edges[u, v][attr] = 0


# Convert NetworkX graph to PyTorch Geometric Data object
node_attrs = ['entity', 'sex', 'origin', 'colour', 'date', 'location', 'course', 'distance', 'condition', 'class_']
edge_attrs = ['relation', 'weight', 'draw']
normalize_node_attributes(G, node_attrs)
normalize_edge_attributes(G, edge_attrs)
data = from_networkx(G)


class RacePlacementGNN(torch.nn.Module):
    def __init__(self):
        super(RacePlacementGNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = torch.nn.Linear(16, 8)
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


model = RacePlacementGNN()


# Create a DataLoader for batching
loader = DataLoader([data], batch_size=1, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(100):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)  # Assuming 'y' is the target race placement
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

