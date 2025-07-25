import dgl
from dgl.base import NID, EID
from dgl.dataloading.neighbor_sampler import BlockSampler
import torch


class NeighborSampler(BlockSampler):

    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # seed_nodes: tensor
        output_nodes = seed_nodes
        new_seed_nodes = torch.unique(seed_nodes)
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                new_seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[EID]
            block = dgl.to_block(frontier, new_seed_nodes)
            block.edata[EID] = eid
            new_seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return new_seed_nodes, output_nodes, blocks
