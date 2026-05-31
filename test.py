from functools import partial
import os
import yaml
import json
from argparse import ArgumentParser
import cv2
import numpy as np
parser = ArgumentParser()
parser.add_argument('--config',
                    default="/home/tien.nguyen/workspace/project/relationformer/configs/info_box_and_windoor_linking.yaml",
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--checkpoint', default="/home/tien.nguyen/workspace/project/relationformer/trained_weights/runs/baseline_info_box_and_windoor_linking_no_num_edges_val_link_f1_center_pred_nodes_focal_loss_10/models/checkpoint_key_metric=0.7936.pt", help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--edge-score-threshold', default=0.,
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[7],
                        help='list of index where skip conn will be made.')


VIS_RESULTS_DIR = "/home/tien.nguyen/workspace/project/relationformer/vis_results"


def draw_graph(
    image,
    nodes,
    edges,
    edge_scores=None,
    save_path=None,
    node_color=(0, 255, 0),
    edge_color=(0, 0, 255),
):
    """
    nodes: Nx2 or Nx4
    edges: Mx2
    edge_scores: list/array of length M
    """

    img = image.copy()

    centers = []

    #
    # Draw nodes
    #
    for idx, node in enumerate(nodes):

        # CASE 1: node = [x, y]
        if len(node) == 2:

            x, y = int(node[0]), int(node[1])

            cv2.circle(img, (x, y), 5, node_color, -1)

            cv2.putText(
                img,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                node_color,
                1,
            )

            centers.append((x, y))

        # CASE 2: node = [x1, y1, x2, y2]
        elif len(node) == 4:

            x1, y1, x2, y2 = map(int, node)

            cv2.rectangle(img, (x1, y1), (x2, y2), node_color, 2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers.append((cx, cy))

            cv2.putText(
                img,
                str(idx),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                node_color,
                1,
            )

    #
    # Draw edges
    #
    for edge_idx, edge in enumerate(edges):

        src, dst = int(edge[0]), int(edge[1])

        if src >= len(centers) or dst >= len(centers):
            continue

        pt1 = centers[src]
        pt2 = centers[dst]

        #
        # Draw line
        #
        cv2.line(
            img,
            pt1,
            pt2,
            edge_color,
            2,
        )

        #
        # Draw score
        #
        # if edge_scores is not None and edge_idx < len(edge_scores):

        #     score = float(edge_scores[edge_idx])

        #     mx = int((pt1[0] + pt2[0]) / 2)
        #     my = int((pt1[1] + pt2[1]) / 2)

        #     text = f"{score:.2f}"

        #     # background for readability
        #     (tw, th), _ = cv2.getTextSize(
        #         text,
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         1,
        #     )

        #     cv2.rectangle(
        #         img,
        #         (mx - 2, my - th - 4),
        #         (mx + tw + 2, my + 2),
        #         (255, 255, 255),
        #         -1,
        #     )

        #     cv2.putText(
        #         img,
        #         text,
        #         (mx, my),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 0, 0),
        #         1,
        #     )

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return img

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def ensure_format(bboxes):
    boxes_new = []
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        
        # to take care of horizontal and vertical edges
        if bbox[2]-bbox[0]<0.2:
            bbox[0] = bbox[0]-0.075
            bbox[2] = bbox[2]+0.075
        if bbox[3]-bbox[1]<0.2:
            bbox[1] = bbox[1]-0.075
            bbox[3] = bbox[3]+0.075
            
        boxes_new.append(bbox)
    return np.array(boxes_new)

def test(args):
    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from dataset_road_network import build_road_network_data
    from models import build_model
    from inference import relation_infer
    from metric_linking import LinkingF1Metric
    from metric_map import BBoxEvaluator
    from box_ops_2D import box_cxcywh_to_xyxy_np
    from utils import image_graph_collate_road_network
    from metrics.topo import compute_topo

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    net = build_model(config).to(device)

    test_ds = build_road_network_data(
        config, mode='test'
    )

    test_loader = DataLoader(test_ds,
                            batch_size=config.DATA.TEST_BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=partial(image_graph_collate_road_network, return_class=getattr(config.MODEL, "GIVEN_BOXES", False)),
                            pin_memory=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # init metric
    # metric = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    metric_f1 = LinkingF1Metric()
    f1_results = []

    metric_node_map = BBoxEvaluator(['node'], max_detections=100)
    metric_edge_map = BBoxEvaluator(['edge'], max_detections=100)

    topo_results = []
    with torch.no_grad():
        print('Started processing test set.')
        for i, batchdata in enumerate(tqdm(test_loader)):

            # extract data and put to device
            images, segs, nodes, edges, classes, org_images = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4], batchdata[5]
            images = images.to(args.device,  non_blocking=False)
            segs = segs.to(args.device,  non_blocking=False)
            nodes = [node.to(args.device,  non_blocking=False) for node in nodes]
            edges = [edge.to(args.device,  non_blocking=False) for edge in edges]

            h, out, _ = net(images, nodes, classes, seg=False)
            pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True,  given_boxes=config.MODEL.GIVEN_BOXES
            )

            os.makedirs(VIS_RESULTS_DIR, exist_ok=True)

            for b_idx in range(len(org_images)):

                # convert tensor image -> numpy
                img = np.array(org_images[b_idx].resize(config.DATA.IMG_SIZE))

                # RGB -> BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # predicted graph
                edge_mask = (
                    pred_edges_box_score[b_idx]
                    > args.edge_score_threshold
                )
                pred_node_np = pred_nodes[b_idx].cpu().numpy()*config.DATA.IMG_SIZE[0]
                pred_edge_np = pred_edges[b_idx][edge_mask]
                pred_edge_score_np = pred_edges_box_score[b_idx][edge_mask]

                vis_img = draw_graph(
                    img,
                    pred_node_np,
                    pred_edge_np,
                    pred_edge_score_np,
                )

                save_path = f"{VIS_RESULTS_DIR}/sample_{i*config.DATA.TEST_BATCH_SIZE+b_idx}.jpg"

                cv2.imwrite(save_path, vis_img)

                f1 = metric_f1([edges[b_idx].cpu().numpy()], [pred_edge_np])
                f1_results.append(f1)

            # Add elements of current batch elem to node map evaluator
            metric_node_map.add(
                pred_boxes=[box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for nodes_ in pred_nodes],
                pred_classes=pred_nodes_box_class,
                pred_scores=pred_nodes_box_score,
                gt_boxes=[box_cxcywh_to_xyxy_np(np.concatenate([nodes_.cpu().numpy(), np.ones_like(nodes_.cpu()) * 0.2], axis=1)) for nodes_ in nodes],
                gt_classes=[np.ones((nodes_.shape[0],)) for nodes_ in nodes]
            )

            # Add elements of current batch elem to edge map evaluator
            pred_edges_box = []
            for edges_, nodes_ in zip(pred_edges, pred_nodes):
                nodes_ = nodes_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                pred_edges_box.append(edges_box)

            gt_edges_box = []
            for edges_, nodes_ in zip(edges, nodes):
                nodes_ , edges_ = nodes_.cpu().numpy(), edges_.cpu().numpy()
                edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
                gt_edges_box.append(edges_box)

            metric_edge_map.add(
                pred_boxes=pred_edges_box,
                pred_classes=pred_edges_box_class,
                pred_scores=pred_edges_box_score,
                gt_boxes=gt_edges_box,
                gt_classes=[np.ones((edges_.shape[0],)) for edges_ in edges]
            )
            
            for node_, edge_, pred_node_, pred_edge_ in zip(nodes, edges, pred_nodes, pred_edges):
                topo_results.append(compute_topo(node_.cpu(), edge_.cpu(), pred_node_, pred_edge_, config.DATA.IMG_SIZE))
    
    # Determine smd
    smd_mean = torch.tensor(f1_results).mean().item()
    smd_std = torch.tensor(f1_results).std().item()
    print(f'F1 value: mean {smd_mean}, std {smd_std}\n')

    # Determine node box ap / ar
    node_metric_scores = metric_node_map.eval()
    print(f"node mAP_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node AP_IoU_0.10_MaxDet_100 {node_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"node AP_IoU_0.20_MaxDet_100 {node_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"node AP_IoU_0.30_MaxDet_100 {node_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"node AP_IoU_0.40_MaxDet_100 {node_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"node AP_IoU_0.50_MaxDet_100 {node_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"node AP_IoU_0.60_MaxDet_100 {node_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"node AP_IoU_0.70_MaxDet_100 {node_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"node AP_IoU_0.80_MaxDet_100 {node_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"node AP_IoU_0.90_MaxDet_100 {node_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    print(f"node mAR_IoU_0.50_0.95_0.05_MaxDet_100 {node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"node AR_IoU_0.10_MaxDet_100 {node_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"node AR_IoU_0.20_MaxDet_100 {node_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"node AR_IoU_0.30_MaxDet_100 {node_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"node AR_IoU_0.40_MaxDet_100 {node_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"node AR_IoU_0.50_MaxDet_100 {node_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"node AR_IoU_0.60_MaxDet_100 {node_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"node AR_IoU_0.70_MaxDet_100 {node_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"node AR_IoU_0.80_MaxDet_100 {node_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"node AR_IoU_0.90_MaxDet_100 {node_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

    # Determine edge box ap / ar
    edge_metric_scores = metric_edge_map.eval()
    print(f"edge mAP_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge AP_IoU_0.10_MaxDet_100 {edge_metric_scores['AP_IoU_0.10_MaxDet_100']}")
    print(f"edge AP_IoU_0.20_MaxDet_100 {edge_metric_scores['AP_IoU_0.20_MaxDet_100']}")
    print(f"edge AP_IoU_0.30_MaxDet_100 {edge_metric_scores['AP_IoU_0.30_MaxDet_100']}")
    print(f"edge AP_IoU_0.40_MaxDet_100 {edge_metric_scores['AP_IoU_0.40_MaxDet_100']}")
    print(f"edge AP_IoU_0.50_MaxDet_100 {edge_metric_scores['AP_IoU_0.50_MaxDet_100']}")
    print(f"edge AP_IoU_0.60_MaxDet_100 {edge_metric_scores['AP_IoU_0.60_MaxDet_100']}")
    print(f"edge AP_IoU_0.70_MaxDet_100 {edge_metric_scores['AP_IoU_0.70_MaxDet_100']}")
    print(f"edge AP_IoU_0.80_MaxDet_100 {edge_metric_scores['AP_IoU_0.80_MaxDet_100']}")
    print(f"edge AP_IoU_0.90_MaxDet_100 {edge_metric_scores['AP_IoU_0.90_MaxDet_100']}\n")

    print(f"edge mAR_IoU_0.50_0.95_0.05_MaxDet_100 {edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100']}")
    print(f"edge AR_IoU_0.10_MaxDet_100 {edge_metric_scores['AR_IoU_0.10_MaxDet_100']}")
    print(f"edge AR_IoU_0.20_MaxDet_100 {edge_metric_scores['AR_IoU_0.20_MaxDet_100']}")
    print(f"edge AR_IoU_0.30_MaxDet_100 {edge_metric_scores['AR_IoU_0.30_MaxDet_100']}")
    print(f"edge AR_IoU_0.40_MaxDet_100 {edge_metric_scores['AR_IoU_0.40_MaxDet_100']}")
    print(f"edge AR_IoU_0.50_MaxDet_100 {edge_metric_scores['AR_IoU_0.50_MaxDet_100']}")
    print(f"edge AR_IoU_0.60_MaxDet_100 {edge_metric_scores['AR_IoU_0.60_MaxDet_100']}")
    print(f"edge AR_IoU_0.70_MaxDet_100 {edge_metric_scores['AR_IoU_0.70_MaxDet_100']}")
    print(f"edge AR_IoU_0.80_MaxDet_100 {edge_metric_scores['AR_IoU_0.80_MaxDet_100']}")
    print(f"edge AR_IoU_0.90_MaxDet_100 {edge_metric_scores['AR_IoU_0.90_MaxDet_100']}\n")

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)