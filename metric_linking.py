import torch
from ignite.metrics import Metric
from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

reinit__is_reduced, _ = optional_import(
    "ignite.metrics.metric",
    IgniteInfo.OPT_IMPORT_VERSION,
    min_version,
    "reinit__is_reduced",
)


class LinkingAccuracyMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def normalize_edge(self, edge):
        return tuple(sorted(edge))

    def edge_to_set(self, edges):

        # tensor -> list
        if torch.is_tensor(edges):
            edges = edges.detach().cpu().tolist()

        return set(
            self.normalize_edge(edge)
            for edge in edges
        )

    def __call__(self, gt_edges, pred_edges):

        batch_size = len(gt_edges)

        for b in range(batch_size):

            gt_set = self.edge_to_set(gt_edges[b])
            pred_set = self.edge_to_set(pred_edges[b])

            correct = len(gt_set & pred_set)
            total = len(pred_set)

            self.correct += correct
            self.total += total

        if self.total == 0:
            return torch.tensor(0.0)

        return torch.tensor(self.correct / self.total)

    def aggregate(self):
        if self.total == 0:
            return 0.0

        return self.correct / self.total


class MeanLinkingAccuracy(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.metric_fn = LinkingAccuracyMetric()
        super().__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.metric_fn.reset()

    @reinit__is_reduced
    def update(self, output):
        gt_edges, pred_edges = output
        return self.metric_fn(gt_edges, pred_edges)

    def compute(self):
        result = self.metric_fn.aggregate()
        self._is_reduced = True
        return result