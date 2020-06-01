import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.scan2cap_dataset import Scan2CapDataset
from lib.solver_captioning import SolverCaptioning
from models.scan2cap_model import Scan2CapModel


from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
VOCABULARY = json.load(open(os.path.join(CONF.PATH.DATA, "vocabulary.json"), "r"))
VOCABULARY = ["<end>"] + VOCABULARY

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = Scan2CapDataset(
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        vocabulary=VOCABULARY,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=augment
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    return dataset, dataloader


def get_model(args):
    with open(GLOVE_PICKLE, "rb") as f:
        glove = pickle.load(f)
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)
    model = Scan2CapModel(vocab_list=VOCABULARY, embedding_dict=glove, feature_channels=input_channels).cuda()
    del glove
    return model


def get_num_params(model):
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataloader, stamp):
    model = get_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    solver = SolverCaptioning(model, DC, dataloader, optimizer, stamp, args.val_step, early_stopping=args.es)
    num_params = get_num_params(model)

    return solver, num_params


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


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


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True)

    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False)

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.tag: stamp += "_" + args.tag.upper()
    root = os.path.join(CONF.PATH.OUTPUT, stamp)
    os.makedirs(root, exist_ok=True)
    solver, num_params = get_solver(args, dataloader, stamp)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=200)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=1)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2500)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=0.0)
    parser.add_argument("--es", type=float, help="early stop", default=-1)
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_augment', action='store_true', help='Do NOT use augmentation in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    train(args)

