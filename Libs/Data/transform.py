from itertools import combinations

import community  # Install with: pip install python-louvain
import networkx as nx
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (add_self_loops, degree, remove_self_loops,
                                   to_dense_adj, to_networkx)


class NodewiseHeuristicProfile(BaseTransform):
    def __call__(self, data: Data) -> Data:
        # Ensure the edge_index is in COO format with self loops
        edge_index, _ = add_self_loops(
            data.edge_index, num_nodes=data.num_nodes)

        # Compute node degree
        deg = degree(edge_index[0], dtype=torch.float)

        # Convert to NetworkX graph to compute clustering coefficient and centrality
        edge_index_np = edge_index.cpu().numpy()
        adj_sparse = sp.coo_matrix((torch.ones(edge_index_np.shape[1]),
                                    (edge_index_np[0], edge_index_np[1])),
                                   shape=(data.num_nodes, data.num_nodes))
        G = nx.from_scipy_sparse_array(adj_sparse)

        # Compute clustering coefficient
        clustering = torch.tensor(
            list(nx.clustering(G).values()), dtype=torch.float)

        # Compute centrality
        centrality = torch.tensor(
            list(nx.degree_centrality(G).values()), dtype=torch.float)

        # Compute assortativity coefficient
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = torch.full((len(G),), assortativity, dtype=torch.float)

        # Compute community structure using the Louvain method
        partition = community.best_partition(G)
        community_labels = torch.tensor(
            [partition[node] for node in range(len(G))])

        # Concatenate features and update data object
        data.h_info_node = torch.stack([
            deg,
            # clustering,
            centrality,
            # assortativity,
            community_labels,
        ], dim=1)

        return data


class LinkwiseHeuristicProfile(object):
    def __call__(self, data):
        # Convert PyG data object to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # 计算边的 betweenness centrality
        edge_bc = nx.edge_betweenness_centrality(G, normalized=True)
        edge_bc_values = torch.tensor(
            [edge_bc[edge] for edge in G.edges()], dtype=torch.float)
        # 计算边的 Jaccard coefficient
        jaccard_values = torch.tensor([self._jaccard_coefficient(
            G, edge) for edge in G.edges()], dtype=torch.float)
        heuristic_edge_attr = torch.stack(
            [edge_bc_values, jaccard_values], dim=1)

        # Update data object
        if data.edge_attr is not None:
            data.edge_attr = torch.cat(
                [data.edge_attr, heuristic_edge_attr], dim=1)
        else:
            data.edge_attr = heuristic_edge_attr

        return data

    def _jaccard_coefficient(self, G, edge):
        set1 = set(G.neighbors(edge[0]))
        set2 = set(G.neighbors(edge[1]))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0


class GraphwiseHeuristicProfile(object):
    def __call__(self, data):
        # Convert PyG data object to NetworkX graph
        G = to_networkx(data, to_undirected=True)

        # Remove self loops as NetworkX shortest path methods do not work with self loops
        G.remove_edges_from(nx.selfloop_edges(G))

        density = torch.tensor(nx.density(G))
        clustering_coefficient = torch.tensor(nx.average_clustering(G))

        # Convert to tensors
        heuristic_graph_attr = torch.stack([
            density,
            clustering_coefficient
        ])

        # Update data object
        data.h_info_g = heuristic_graph_attr

        return data
