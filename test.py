import argparse
import cv2
import numpy as np
import os
import torch
import pdb

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar, keep_bbox_from_lidar_range_v2, vis_pc_plotly
from model import PointPillars
from dataset import Kitti, Custom


def point_range_filter(pts, point_range=[-1, -40, -3, 70.4, 40, 3]):
    '''
    pts: [points]
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts


def main(args):
    if args.dataset_name == 'kitti':
        CLASSES = Kitti.CLASSES
    elif args.dataset_name == 'custom':
        CLASSES = Custom.CLASSES
    else:
        raise ValueError("Dataset name should be 'kitti' or 'custom'")

    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([-1, -40, -3, 70.4, 40, 3], dtype=np.float32)                #[0, -40, -3, 70.4, 40, 0.0]

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model_state_dict'])
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError 
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
        
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]

    result_filter = keep_bbox_from_lidar_range_v2(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    #vis_pc(pc, bboxes=lidar_bboxes, labels=labels)
    vis_pc_plotly(pc, bboxes=lidar_bboxes, labels=labels)

    if gt_label is not None:
        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_lidar = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        bboxes_lidar = bboxes_lidar[sel]

        gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, bboxes_lidar], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        #vis_pc(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)
        vis_pc_plotly(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--dataset_name', default='custom', help='your dataset name')
    parser.add_argument('--pc_path', default='', help='your point cloud path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    main(args)
