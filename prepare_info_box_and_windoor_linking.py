from pathlib import Path
import json
import shutil
import warnings
import random

import numpy as np
import pyvista as pv
from PIL import Image


INFO_CATEGORY_IDS = [1]
DOOR_CATEGORY_IDS = [2, 3]
WINDOW_CATEGORY_IDS = [4, 5]

IS_TEST = True
TRAIN_RATIO = 0.9
RANDOM_SEED = 42


def coco_bbox_to_center(bbox, image_w, image_h):
    """
    COCO bbox format:
    [x, y, w, h]
    """
    x, y, w, h = bbox

    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h

    return cx, cy, w / image_w, h / image_h


def create_split_dirs(root_dir):
    root_dir = Path(root_dir)

    dirs = {}

    for split in ["train", "val", "test"]:

        split_dir = root_dir / split

        dirs[split] = {
            "raw": split_dir / "raw",
            "seg": split_dir / "seg",
            "vtp": split_dir / "vtp",
        }

        for d in dirs[split].values():
            d.mkdir(parents=True, exist_ok=True)

    return dirs


def collect_samples(coco_root):
    coco_root = Path(coco_root)

    samples = []

    annotation_files = sorted(
        coco_root.glob("*/task*/annotations/instances_default.json")
    )

    for ann_file in annotation_files:

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

        annotations_by_image = {}

        for ann in coco["annotations"]:
            annotations_by_image.setdefault(
                ann["image_id"], []
            ).append(ann)

        image_dir = ann_file.parent.parent / "images"

        for image_id, anns in annotations_by_image.items():

            image_info = images[image_id]

            image_path = image_dir / image_info["file_name"]

            if not image_path.exists():
                print(f"Missing image: {image_path}")
                continue

            samples.append(
                {
                    "image_info": image_info,
                    "image_path": image_path,
                    "anns": anns,
                }
            )

    return samples


def process_sample(
    sample,
    sample_name,
    output_dirs,
):
    image_info = sample["image_info"]
    anns = sample["anns"]
    image_path = sample["image_path"]

    image_w = image_info["width"]
    image_h = image_info["height"]

    #
    # Build nodes
    #
    nodes = []
    node_id_map = {}

    #
    # Store mapping:
    # ("window", "7") -> node_idx
    #
    object_id_to_node = {}

    for ann in anns:

        cat_id = ann["category_id"]

        if cat_id not in (
            WINDOW_CATEGORY_IDS
            + DOOR_CATEGORY_IDS
            + INFO_CATEGORY_IDS
        ):
            continue

        bbox = ann["bbox"]

        cx, cy, nw, nh = coco_bbox_to_center(
            bbox,
            image_w,
            image_h,
        )

        if cat_id in WINDOW_CATEGORY_IDS:
            cls = 0
        elif cat_id in DOOR_CATEGORY_IDS:
            cls = 1
        else:
            cls = 2

        node_idx = len(nodes)

        nodes.append(
            {
                "cx": cx,
                "cy": cy,
                "w": nw,
                "h": nh,
                "cls": cls,
            }
        )

        node_id_map[ann["id"]] = node_idx

        attrs = ann.get("attributes", {})

        #
        # register window / door ids
        #
        if cat_id in WINDOW_CATEGORY_IDS:

            window_id = str(
                attrs.get("window_id", "")
            ).strip()

            if window_id:
                object_id_to_node[
                    ("window", window_id)
                ] = node_idx

        elif cat_id in DOOR_CATEGORY_IDS:

            door_id = str(
                attrs.get("door_id", "")
            ).strip()

            if door_id:
                object_id_to_node[
                    ("door", door_id)
                ] = node_idx

    #
    # Build edges
    #
    edges = []
    info_ann_count = 0

    for ann in anns:

        if ann["category_id"] not in INFO_CATEGORY_IDS:
            continue

        info_ann_count += 1

        info_node_idx = node_id_map[ann["id"]]

        attrs = ann.get("attributes", {})

        window_id = str(
            attrs.get("window_id", "")
        ).replace(" ", "")

        door_id = str(
            attrs.get("door_id", "")
        ).replace(" ", "")

        #
        # link to window
        #
        if window_id:

            key = ("window", window_id)

            if key in object_id_to_node:

                target_idx = object_id_to_node[key]

                edges.append(
                    [target_idx, info_node_idx]
                )

        #
        # link to door
        #
        if door_id:

            key = ("door", door_id)

            if key in object_id_to_node:

                target_idx = object_id_to_node[key]

                edges.append(
                    [target_idx, info_node_idx]
                )

    #
    # skip empty graphs
    #
    if len(nodes) == 0:
        return False

    if len(edges) < info_ann_count / 2 or info_ann_count == 0:
        warnings.warn(
            f"Got {info_ann_count} information annotations "
            f"but only {len(edges)} edges."
        )
        return False

    #
    # Save raw image
    #
    raw_image_path = (
        output_dirs["raw"] / f"{sample_name}_data.png"
    )

    shutil.copy(image_path, raw_image_path)

    #
    # Save segmentation placeholder
    #
    seg_image = np.zeros(
        (image_h, image_w),
        dtype=np.uint8,
    )

    seg_path = (
        output_dirs["seg"] / f"{sample_name}_seg.png"
    )

    Image.fromarray(seg_image).save(seg_path)

    #
    # Build VTP graph
    #
    points = np.array(
        [
            [n["cx"], n["cy"], 0.0]
            for n in nodes
        ],
        dtype=np.float32,
    )

    poly = pv.PolyData(points)

    #
    # edge format:
    # [2, src, dst]
    #
    if len(edges) > 0:

        line_array = []

        for src, dst in edges:
            line_array.extend([2, src, dst])

        poly.lines = np.array(
            line_array,
            dtype=np.int64,
        )

    #
    # save node features
    #
    poly.point_data["width"] = np.array(
        [n["w"] for n in nodes],
        dtype=np.float32,
    )

    poly.point_data["height"] = np.array(
        [n["h"] for n in nodes],
        dtype=np.float32,
    )

    poly.point_data["class"] = np.array(
        [n["cls"] for n in nodes],
        dtype=np.int32,
    )

    vtp_path = (
        output_dirs["vtp"] / f"{sample_name}_graph.vtp"
    )

    poly.save(vtp_path)

    print(
        f"[OK] {sample_name} "
        f"nodes={len(nodes)} "
        f"edges={len(edges)}"
    )

    return True


def build_graph_from_coco(
    coco_root,
    output_root,
):
    output_dirs = create_split_dirs(output_root)

    #
    # Collect all samples first
    #
    samples = collect_samples(coco_root)

    print(f"Collected {len(samples)} samples")

    #
    # Shuffle
    #
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    #
    # Train-val split
    #
    if IS_TEST:

        split_data = {
            "test": samples,
        }

        split_counters = {
            "test": 0,
        }

        print(f"Test samples: {len(samples)}")

    else:

        split_idx = int(len(samples) * TRAIN_RATIO)

        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")

        split_data = {
            "train": train_samples,
            "val": val_samples,
        }

        split_counters = {
            "train": 0,
            "val": 0,
        }

    #
    # Process
    #
    for split_name, split_samples in split_data.items():

        print(f"\nProcessing {split_name}...")

        for sample in split_samples:

            sample_idx = split_counters[split_name]

            sample_name = f"sample_{sample_idx:06d}"

            success = process_sample(
                sample=sample,
                sample_name=sample_name,
                output_dirs=output_dirs[split_name],
            )

            if success:
                split_counters[split_name] += 1

    print("\nDone.")

    for split_name, count in split_counters.items():
        print(f"Final {split_name} samples: {count}")


if __name__ == "__main__":

    build_graph_from_coco(
        coco_root="/home/tien.nguyen/workspace/project/common/data/yap-2/cropped_dataset_20260528/train_val",
        output_root="/home/tien.nguyen/workspace/project/relationformer/data/info_box_and_windoor_linking",
    )