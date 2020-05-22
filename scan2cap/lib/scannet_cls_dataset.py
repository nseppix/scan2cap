'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")

# MAX_DES_LEN = 30
# MAX_DES_LEN = 117

class ScannetPretrainDataset(Dataset):

    def __init__(self, scanrefer, scanrefer_all_scene,
                 split="train",
                 num_points=40000,
                 use_height=False,
                 use_color=False,
                 use_normal=False,
                 use_multiview=False,
                 augment=False):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.augment = augment

        # load data
        self._load_data()
        self.multiview_data = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.samples[idx]["scene_id"]
        object_id = int(self.samples[idx]["object_id"])

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        bbox = instance_bboxes[instance_bboxes[:, 7] == object_id, :]
        class_label = bbox[0, 6]

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        pcl_color = pcl_color[choices]

        target_bboxes = bbox[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along X-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32)  # point cloud data including features
        data_dict["ref_center_label"] = target_bboxes[0, :3].astype(np.float32)
        data_dict["ref_size_residual_label"] = target_bboxes[0, 3:6].astype(np.float32)
        data_dict["ref_nyu40_label"] = np.array(int(class_label)).astype(np.int64)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _load_data(self):
        print("loading data...")

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        self.samples = []

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_vert.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_bbox.npy")

            samples = [{"scene_id": scene_id, "object_id": self.scene_data[scene_id]["instance_bboxes"][i, 7], "class_label": self.scene_data[scene_id]["instance_bboxes"][i, 6]} for i in range(self.scene_data[scene_id]["instance_bboxes"].shape[0])]
            self.samples.extend(samples)

        # # load multiview database
        # if self.use_multiview:
        #     self.multiview_database = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
