import os.path as osp
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from Libs.Data.gene_cas import config
from Libs.Exp.config import search_args
import torch_geometric.transforms as T
from Libs.Data.transform import (GraphwiseHeuristicProfile,
                                 LinkwiseHeuristicProfile,
                                 NodewiseHeuristicProfile)
from torch_geometric.loader import DataLoader


class InfoCas(InMemoryDataset):
    r"""A PyTorch Geometric dataset representing cascading information networks.

    The dataset is created from raw text files representing the cascades of information,
    with nodes representing entities and edges representing the propagation of information.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (e.g., 'weibo', 'twitter', etc.).
        split (str, optional): The type of split ('train', 'val', 'test'). (default: 'train')
        max_seq (int, optional): The maximum sequence length. (default: 100)
        transform (callable, optional): A function/transform to apply to the data object.
        pre_transform (callable, optional): A function/transform to apply to the data object before saving to disk.

    Attributes:
        observation_time (int): The observation time derived from the `observation_time_dict` based on the dataset name.
    """

    def __init__(self, root, name, split='train', max_seq=100, mode='continuous', transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.split = split
        self.max_seq = max_seq
        self.mode = mode
        self.observation_time = config[self.name]["Observation Time"]
        assert split in ['train', 'val', 'test', 'full']
        path = self.processed_paths[[
            'train', 'val', 'test', 'full'].index(split)]
        super(InfoCas, self).__init__(root, transform, pre_transform)
        self.data = torch.load(path)
        self.diffusion_graph = torch.load(
            osp.join(self.processed_dir, 'diffusion.pt'))

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['train.txt', 'val.txt', 'test.txt', 'full.txt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt', 'full.pt']

    def download(self):
        pass

    def process(self):
        import os
        # 检查文件是否已存在
        if os.path.exists(self.processed_paths[['train', 'val', 'test', 'full'].index(self.split)]):
            return
        data_list = []
        graphs = self._sequence2list(
            self.raw_paths[['train', 'val', 'test', 'full'].index(self.split)])
        labels = self._read_labels(
            self.raw_paths[['train', 'val', 'test', 'full'].index(self.split)])
        self._generate_diffusion_graph(graphs, labels, self.observation_time)
        if self.mode == 'continuous':
            data_list.extend(self._generate_data_continuous(
                graphs, labels, self.observation_time))
        else:
            data_list.extend(self._generate_data_discrete(
                graphs, labels, self.observation_time))
        # data_list = self.collate(data_list)
        # data, slices = self.collate(data_list)
        torch.save(data_list, self.processed_paths[['train', 'val', 'test', 'full'].index(self.split)])
        # torch.save((data, slices), self.processed_paths[['train', 'val', 'test', 'full'].index(self.split)])

    def _sequence2list(self, filename):
        r"""Converts raw data in `filename` to a list of graphs.

        Args:
            filename (str): The name of the file containing the raw data.

        Returns:
            Dict: A dictionary where each key is a unique identifier of a graph, and each value is a list representing the graph.
        """
        graphs = {}
        with open(filename, 'r') as f:
            for line in f:
                paths = line.strip().split('\t')[:-1][:self.max_seq + 1]
                graphs[paths[0]] = []
                for i in range(1, len(paths)):
                    nodes = paths[i].split(":")[0]
                    time = paths[i].split(":")[1]
                    graphs[paths[0]].append(
                        [[int(x) for x in nodes.split(",")], int(time)])

        return graphs

    def _read_labels(self, filename):
        r"""Reads labels from `filename`.

        Args:
            filename (str): The name of the file containing the labels.

        Returns:
            Dict: A dictionary where each key is a unique identifier of a graph, and each value is the label of the graph.
        """
        labels = {}
        with open(filename, 'r') as f:
            for line in f:
                id = line.strip().split('\t')[0]
                labels[id] = line.strip().split('\t')[-1]
        return labels

    def _generate_diffusion_graph(self, graphs, labels, observation_time, weight=True):
        r"""Generates a list of torch_geometric.data.Data objects from raw graphs and labels.

        Args:
            graphs (Dict): A dictionary where each key is a unique identifier of a graph, and each value is a list representing the graph.
            labels (Dict): A dictionary where each key is a unique identifier of a graph, and each value is the label of the graph.
            observation_time (int): The observation time.
            weight (bool, optional): Whether to add weights to the edges. (default: True)

        Returns:
            List: A list of torch_geometric.data.Data objects.
        """
        diffusion_graph = nx.DiGraph()
        for key, cascades in graphs.items():
            list_edge = []
            times = []
            t_o = observation_time

            # add edges into graph
            for cascade in cascades:
                t = cascade[1]
                if t >= t_o:
                    continue
                nodes = cascade[0]
                if len(nodes) == 1:
                    diffusion_graph.add_nodes_from(
                        [(nodes[0], {"id": int(nodes[0])})])
                    times.append(1)
                    continue
                nodes_with_ids = [
                    (nodes[-1], {"id": int(nodes[-1])}), (nodes[-2], {"id": int(nodes[-2])})]
                diffusion_graph.add_nodes_from(nodes_with_ids)
                if weight:
                    edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                    times.append(1 - t / t_o)
                else:
                    edge = (nodes[-1], nodes[-2])
                list_edge.append(edge)
            if weight:
                diffusion_graph.add_weighted_edges_from(
                    list_edge, weight='edge_weight')
            else:
                diffusion_graph.add_edges_from(list_edge)

            if diffusion_graph.number_of_nodes() <= 1:
                continue

        data = from_networkx(diffusion_graph)
        torch.save(data, osp.join(self.processed_dir, 'diffusion.pt'))

    def _generate_data_discrete(self, graphs, labels, observation_time, weight=True):
        r"""Generates a list of torch_geometric.data.Data objects from raw graphs and labels.

        Args:
            graphs (Dict): A dictionary where each key is a unique identifier of a graph, and each value is a list representing the graph.
            labels (Dict): A dictionary where each key is a unique identifier of a graph, and each value is the label of the graph.
            observation_time (int): The observation time.
            weight (bool, optional): Whether to add weights to the edges. (default: True)

        Returns:
            List: A list of torch_geometric.data.Data objects.
        """
        data_list = []
        for key, cascades in tqdm(graphs.items()):
            y = int(labels[key])
            list_edge = []
            times = []
            t_o = observation_time
            snapshots = []
            # add edges into graph
            for cascade in cascades:
                snapshot = nx.DiGraph()
                t = cascade[1]
                if t >= t_o:
                    continue
                nodes = cascade[0]
                if len(nodes) == 1:
                    continue
                if weight:
                    edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                    times.append(1 - t / t_o)
                else:
                    edge = (nodes[-1], nodes[-2])
                list_edge.append(edge)
                if weight:
                    snapshot.add_weighted_edges_from(
                        list_edge, weight='edge_weight')
                else:
                    snapshot.add_edges_from(list_edge)

                if snapshot.number_of_nodes() <= 1:
                    continue

                snapshot = from_networkx(snapshot)
                snapshot.y = torch.tensor([y], dtype=torch.float)
                snapshot.cas_id = key
                if self.pre_transform is not None:
                    snapshot = self.pre_transform(snapshot)
                snapshots.append(snapshot)
            data_list.append(snapshots)
        return data_list

    def _generate_data_continuous(self, graphs, labels, observation_time, weight=True):
        r"""Generates a list of torch_geometric.data.Data objects from raw graphs and labels.

        Args:
            graphs (Dict): A dictionary where each key is a unique identifier of a graph, and each value is a list representing the graph.
            labels (Dict): A dictionary where each key is a unique identifier of a graph, and each value is the label of the graph.
            observation_time (int): The observation time.
            weight (bool, optional): Whether to add weights to the edges. (default: True)

        Returns:
            List: A list of torch_geometric.data.Data objects.
        """
        data_list = []
        for key, cascades in tqdm(graphs.items()):
            y = int(labels[key])
            list_edge = []
            times = []
            t_o = observation_time
            g = nx.DiGraph()
            # add edges into graph
            for cascade in cascades:
                t = cascade[1]
                if t >= t_o:
                    continue
                nodes = cascade[0]
                if len(nodes) == 1:
                    times.append(1)
                    g.add_nodes_from([(nodes[0], {"id": int(nodes[0])})])
                    continue
                nodes_with_ids = [
                    (nodes[-1], {"id": int(nodes[-1])}), (nodes[-2], {"id": int(nodes[-2])})]
                g.add_nodes_from(nodes_with_ids)
                if weight:
                    edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                    times.append(1 - t / t_o)
                else:
                    edge = (nodes[-1], nodes[-2])
                list_edge.append(edge)
            if weight:
                g.add_weighted_edges_from(
                    list_edge, weight='edge_weight')
            else:
                g.add_edges_from(list_edge)

            if g.number_of_nodes() <= 1:
                continue

            data = from_networkx(g)
            data.y = torch.tensor([y], dtype=torch.float)
            data.cas_id = key
            data.edge_weight = data.edge_weight.unsqueeze(1)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(size={len(self)}, name={self.name}, split={self.split}, max_seq={self.max_seq}, observation_time={self.observation_time})')


def get_dataloader():
    pre_transform = T.Compose([
        T.ToUndirected(),
        T.AddSelfLoops(),
        # Appends the Local Degree Profile (LDP) to the node features
        T.LocalDegreeProfile(),
        # Heuristic information augmentation
        NodewiseHeuristicProfile(),
        # LinkwiseHeuristicProfile(),
        # GraphwiseHeuristicProfile(),
        # Dimensionality reduction of node features via Singular Value Decomposition (SVD)
        T.SVDFeatureReduction(out_channels=3),
        # Row-normalizes the attributes given in attrs to sum-up to one
        T.NormalizeFeatures(),
    ])
    transform = T.Compose([])
    dataset_root_path = osp.join(
        search_args.root, search_args.dataset_root_path)
    full_dataset = InfoCas(root=dataset_root_path, name=search_args.dataset_name,
                           split='full', pre_transform=pre_transform, transform=transform)
    train_dataset = InfoCas(root=dataset_root_path, name=search_args.dataset_name,
                            split='train', pre_transform=pre_transform, transform=transform)
    val_dataset = InfoCas(root=dataset_root_path, name=search_args.dataset_name,
                          split='val', pre_transform=pre_transform, transform=transform)
    test_dataset = InfoCas(root=dataset_root_path, name=search_args.dataset_name,
                           split='test', pre_transform=pre_transform, transform=transform)

    train_dataset = train_dataset[0]
    val_dataset = val_dataset[0]
    test_dataset = test_dataset[0]
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=search_args.batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=search_args.batch_size,
                            shuffle=False, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=search_args.batch_size,
                             shuffle=False, drop_last=True, num_workers=num_workers)
    d_in = train_dataset[0].num_features
    d_edge_weight = train_dataset[0].edge_weight.unsqueeze(1).shape[1]
    d_hinfo_node = train_dataset[0].h_info_node.shape[1]
    # diff_g = full_dataset.diffusion_graph
    return (train_loader, val_loader, test_loader, d_in, d_edge_weight, d_hinfo_node)
    # return (train_dataset, val_dataset, test_dataset, d_in, d_edge_weight, d_hinfo_node, diff_g)
