import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd()))
from models.scan2cap_model import Scan2CapModel

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.pc_utils import write_ply_rgb
from utils.box_util import get_3d_box
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from lib.scan2cap_dataset import Scan2CapDataset
from models.pointnet_extractor_module import PointNetExtractor

# constants
SCANNET_ROOT = "../data/scannet/scans/" # TODO point this to your scannet data
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
VOCABULARY = ["<end>"] + json.load(open(os.path.join(CONF.PATH.DATA, "vocabulary.json"), "r"))


global_correct = 0
global_total = 0

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = Scan2CapDataset(
        scanrefer=scanrefer,
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return dataset, dataloader

def get_model(args):
    with open(GLOVE_PICKLE, "rb") as f:
        glove = pickle.load(f)
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)
    model = Scan2CapModel(vocab_list=VOCABULARY, embedding_dict=glove, feature_channels=input_channels, use_votenet=args.use_votenet, use_attention=args.use_attention, objectness_thresh=args.objectness_thresh, n_closest=args.n_closest).cuda()
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    # path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model_last.pth")
    model.load_state_dict(torch.load(path), strict=False)
    del glove
    return model

def get_scanrefer(args):
    scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
    all_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.scene_id:
        assert args.scene_id in all_scene_list, "The scene_id is not found"
        scene_list = [args.scene_id]
    else:
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))

    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(bbox, output_file, votenet_bboxes_alphas=None):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=4, slices=4):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0) # 8 x 3

        return corners

    def edges_to_verts_indices(edges, color):
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[c / 255 for c in color] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)

    gt_color = [0, 255, 0]
    edges = get_bbox_edges(box_min, box_max)

    edges_to_verts_indices(edges, gt_color)

    if votenet_bboxes_alphas is not None:
        votenet_bboxes, votenet_alphas = votenet_bboxes_alphas
        for vn_bbox, vn_alpha in list(sorted(zip(votenet_bboxes, votenet_alphas), key=lambda x: x[1], reverse=True))[:16]:
            corners = get_bbox_corners(vn_bbox)
            box_min = np.min(corners, axis=0)
            box_max = np.max(corners, axis=0)
            vn_edges = get_bbox_edges(box_min, box_max)

            # print(vn_alpha)

            color = [int((1-vn_alpha) * 255), 0, int(vn_alpha * 255)]
            edges_to_verts_indices(vn_edges, color)

    write_ply(verts, colors, indices, output_file)

def read_mesh(filename):
    """ read XYZ for each vertex.
    """

    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

    return vertices, plydata['face']

def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")
    
    return PlyData([vertices, faces])

def align_mesh(scene_id):
    vertices, faces = read_mesh(SCANNET_MESH.format(scene_id, scene_id))
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)

    return mesh

def dump_results(args, scanrefer, data, config):
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis")
    os.makedirs(dump_dir, exist_ok=True)

    # from inputs
    ids = data['scan_idx'].detach().cpu().numpy()
    point_clouds = data['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    pcl_color = data["pcl_color"].detach().cpu().numpy()
    if args.use_color:
        pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)
    
    # from network outputs
    # detection
    # ground truth
    gt_center = data['ref_center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_size_residual = data['ref_size_residual_label'].cpu().numpy() # B,K2,3
    # reference
    nyu40_label = data["ref_nyu40_label"].detach().cpu().numpy()
#    prediction = torch.argmax(data["ref_obj_cls_scores"], dim=1).detach().cpu().numpy() + 1
    hypo = data["caption_indices"]
    ref = data["lang_indices"]

    # global global_correct
    # global global_total
    # global_correct += np.sum(nyu40_label == prediction)
    # global_total += batch_size

    # print("NYU40_LABEL", [DC.nyu40id2label[i] for i in list(nyu40_label)])
    # print("PREDICTION", [DC.nyu40id2label[i] for i in list(prediction)])
    # print("ACC", global_correct / global_total)
    # print(torch.max(data["alphas"]))
    # print(torch.mean(torch.sum((data["alphas"][:, 0, :] > 0).to(dtype=torch.float32), dim=1)))

    for i in range(batch_size):
        # basic info
        idx = ids[i]
        scene_id = scanrefer[idx]["scene_id"]
        object_id = scanrefer[idx]["object_id"]
        ann_id = scanrefer[idx]["ann_id"]
    
        # scene_output
        scene_dump_dir = os.path.join(dump_dir, scene_id)
        if not os.path.exists(scene_dump_dir):
            os.mkdir(scene_dump_dir)

            # # Dump the original scene point clouds
            mesh = align_mesh(scene_id)
            mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))

            write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))

        hypo_strings = ' '.join([VOCABULARY[index] for index in hypo[i] if index > 0])
        ref_string = ' '.join([VOCABULARY[index] for index in ref[i] if index > 0])

        print("Ref", ref_string)
        print("Hypo", hypo_strings)

        # visualize the gt reference box
        # NOTE: for each object there should be only one gt reference box
        object_dump_dir = os.path.join(scene_dump_dir, "gt_{}_{}_{}_{}_{}.ply".format(scene_id, object_id, ann_id, DC.nyu40id2label[nyu40_label[i]], hypo_strings))
        gt_obb = np.zeros((7,))
        gt_obb[0:3] = gt_center[i]
        gt_obb[3:6] = gt_size_residual[i]
        gt_bbox = get_3d_box(gt_size_residual[i], 0, gt_center[i])

        if "aggregated_vote_features" in data:
            centers = data["aggregated_vote_xyz"][i].cpu()
            size_classes = torch.argmax(data['size_scores'][i], dim=1, keepdim=True).cpu()
            sizes = torch.gather(data["size_residuals"][i].cpu(), dim=1, index=size_classes.unsqueeze(2).expand(-1, 1, 3)).squeeze(1)
            objectness = torch.softmax(data["objectness_scores"][i], dim=-1)[:,1].cpu()

            sizes = torch.stack([torch.tensor(config.param2obb(center.numpy(), 0, 0, cls.numpy(), size.numpy()))[3:6] for center, cls, size in zip(centers, size_classes, sizes)]).to(dtype=torch.float32)

            # print(centers.shape, size_classes.shape, sizes.shape, objectness.shape)

            bboxes = torch.cat([centers, sizes], dim=1)

            # print(centers[0], sizes[0], bboxes[0], objectness[0])

            #objectness_masks = objectness > .5
            objectness_masks = data["object_mask"][i]
            objectness = objectness[objectness_masks]
            bboxes = bboxes[objectness_masks, :]
           

            votenet_bboxes_alphas = (bboxes.numpy(), objectness.cpu().numpy())
            write_bbox(gt_obb, os.path.join(object_dump_dir)[:-4] + "_votenet.ply", votenet_bboxes_alphas)
            
            if "alphas" in data:
                alphas = data["alphas"][i]
                alphas = alphas[:, objectness_masks]

                for j, alphas_i in enumerate(alphas):
                    if torch.max(alphas_i) == 0:
                        break

                    if not os.path.exists(object_dump_dir):
                        alphas_timestep = (bboxes.numpy(), alphas_i.cpu().numpy())
                        dir = object_dump_dir[:-4] + "_attention"
                        if not os.path.exists(dir):
                            os.mkdir(dir)
                        write_bbox(gt_obb, os.path.join(dir, f"timestep{j:02d}_{VOCABULARY[hypo[i, j]]}.ply"), alphas_timestep)
        else:
            votenet_bboxes_alphas = None

        if not os.path.exists(object_dump_dir):
            write_bbox(gt_obb, os.path.join(object_dump_dir))
        
        


def visualize(args):
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    scanrefer = [s for s in scanrefer if s["scene_id"] in ["scene0064_00",
                                                           "scene0100_00",
                                                           "scene0086_00",
                                                           "scene0081_00",
                                                           "scene0084_00",
                                                           "scene0164_00",
                                                           "scene0193_00",
                                                           "scene0144_00",
                                                           "scene0221_00",
                                                           "scene0203_00",
                                                           "scene0222_00",
                                                           "scene0231_00"]]

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC, False)

    # model
    model = get_model(args)
    model.eval()
    
    # evaluate
    print("visualizing...")
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data)
        
        # visualize
        dump_results(args, scanrefer, data, DC)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_id", type=str, help="scene id", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=2)
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--num_proposals', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_nms', action='store_true', help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_train', action='store_true', help='Use the training set.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    parser.add_argument('--pnextractor_cp', type=str, help="Checkpoint location for pointnet extractor.", default=None)
    parser.add_argument('--decoder_cp', type=str, help="Checkpoint location for LSTM decoder.", default=None)
    parser.add_argument('--use_votenet', action='store_true', help="Use votenet as additional feature extractor. (Required for attention)")
    parser.add_argument('--use_attention', action='store_true', help="Use attention for captioning, only works if votenet is used")
    parser.add_argument('--objectness_thresh', type=float, help="Threshold for accepting objects proposed by votenet", default=.75)
    parser.add_argument('--n_closest', type=int, help="Number of n closest votenet proposals are considered", default=32)
    parser.add_argument('--cp', type=str, help="Checkpoint location for Scan2Cap model.", default=None)
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    visualize(args)
