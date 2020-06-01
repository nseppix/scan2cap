import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from collections import Counter

from lib.config import CONF
from lib.scan2cap_dataset import Scan2CapDataset

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

OUT_FILE = os.path.join(CONF.PATH.DATA, "distribution.json")


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    # randomly choose scenes
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    return new_scanrefer_train, scanrefer_val, all_scene_list

scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
scanrefer = {
    "train": scanrefer_train,
    "val": scanrefer_val
}

split = "train"

dataset = Scan2CapDataset(
    scanrefer=scanrefer[split],
    scanrefer_all_scene=all_scene_list,
    vocabulary=[],
    split=split,
    num_points=1,
    use_height=False,
    use_color=False,
    use_normal=False,
    use_multiview=False,
    lang_tokens=True
)

object_ids = [s["ref_nyu40_label"].item() for s in tqdm(dataset)]
distribution = Counter(object_ids)

json.dump(dict(distribution), open(OUT_FILE, "w+"), indent=4)