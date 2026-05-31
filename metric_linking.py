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


class LinkingF1Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.pred_total = 0
        self.gt_total = 0

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

    def _f1(self, tp, pred_total, gt_total):
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / gt_total if gt_total > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def __call__(self, gt_edges, pred_edges):

        batch_size = len(gt_edges)

        for b in range(batch_size):

            gt_set = self.edge_to_set(gt_edges[b])
            pred_set = self.edge_to_set(pred_edges[b])

            self.tp += len(gt_set & pred_set)
            self.pred_total += len(pred_set)
            self.gt_total += len(gt_set)

        return torch.tensor(self._f1(self.tp, self.pred_total, self.gt_total))

    def aggregate(self):
        return self._f1(self.tp, self.pred_total, self.gt_total)


class MeanLinkingF1(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.metric_fn = LinkingF1Metric()
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