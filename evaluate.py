import argparse
import numpy as np
import os
import torch
import pdb
from tqdm import tqdm

from utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev, keep_bbox_from_lidar_range_v2, \
    write_label_filtered_with_score
from dataset import Kitti, get_dataloader, Custom
from model import PointPillars


def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
    det_results: list,
    gt_results: dict(id -> det_results)
    CLASSES: dict
    '''
    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    # 1. calculate iou
    ious = {
        'bbox_bev': [],
        'bbox_3d': []
    }
    ids = list(sorted(gt_results.keys()))
    for id in ids:
        gt_result = gt_results[id]['annos']
        det_result = det_results[id]

        # 1.1, bev iou
        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)

        gt_bev = np.concatenate([gt_location[:, [0, 2]], gt_dimensions[:, [0, 2]], gt_rotation_y[:, None]], axis=-1)
        det_bev = np.concatenate([det_location[:, [0, 2]], det_dimensions[:, [0, 2]], det_rotation_y[:, None]], axis=-1)
        iou_bev_v = iou_bev(torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda())
        ious['bbox_bev'].append(iou_bev_v.cpu().numpy())

        # 1.2, 3dbboxes iou
        gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        iou3d_v = iou3d_camera(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda())
        ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    MIN_IOUS = {
        'Pedestrian': [0.5, 0.5],
        'Cyclist': [0.5, 0.5],
        'Car': [0.5, 0.5],
        'Wheelchair': [0.5, 0.5]
    }

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]

            # 1. bbox property
            total_gt_ignores, total_det_ignores, total_scores = [], [], []
            for id in ids:
                gt_result = gt_results[id]['annos']
                det_result = det_results[id]

                # 1.1 gt bbox property
                cur_gt_names = gt_result['name']
                gt_ignores, dc_bboxes = [], []
                for j, cur_gt_name in enumerate(cur_gt_names):
                    if cur_gt_name == cls:
                        gt_ignores.append(0)
                    elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                        gt_ignores.append(1)
                    elif cls == 'Car' and cur_gt_name == 'Van':
                        gt_ignores.append(1)
                    else:
                        gt_ignores.append(-1)
                    
                total_gt_ignores.append(gt_ignores)

                # 1.2 det bbox property
                cur_det_names = det_result['name']
                det_ignores = []
                for j, cur_det_name in enumerate(cur_det_names):
                    if cur_det_name == cls:
                        det_ignores.append(0)
                    else:
                        det_ignores.append(1)
                total_det_ignores.append(det_ignores)
                total_scores.append(det_result['score'])

            # 2. calculate scores thresholds for PR curve
            tp_scores = []
            for i, id in enumerate(ids):
                cur_eval_ious = eval_ious[i]
                gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                scores = total_scores[i]

                nn, mm = cur_eval_ious.shape
                assigned = np.zeros((mm, ), dtype=np.bool_)
                for j in range(nn):
                    if gt_ignores[j] == -1:
                        continue
                    match_id, match_score = -1, -1
                    for k in range(mm):
                        if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                            match_id = k
                            match_score = scores[k]
                    if match_id != -1:
                        assigned[match_id] = True
                        if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                            tp_scores.append(match_score)
            
            total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
            score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)
        
            # 3. draw PR curve and calculate mAP
            tps, fns, fps, total_aos = [], [], [], []

            for score_threshold in score_thresholds:
                tp, fn, fp = 0, 0, 0
                #aos = 0
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_iou = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
                                if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                    match_iou = cur_eval_ious[j, k]
                                    match_id = k
                                elif det_ignores[k] == 1 and match_iou == -1:
                                    match_id = k

                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp += 1
                        else:
                            if gt_ignores[j] == 0:
                                fn += 1
                        
                    for k in range(mm):
                        if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                            fp += 1
                    
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)

            tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

            recalls = tps / (tps + fns)
            precisions = tps / (tps + fps)
            for i in range(len(score_thresholds)):
                precisions[i] = np.max(precisions[i:])
            
            sums_AP = 0
            for i in range(0, len(score_thresholds), 4):
                sums_AP += precisions[i]
            mAP = sums_AP / 11 * 100
            eval_ap_results[cls].append(mAP)

        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f}', file=f)
        
        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
    
    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f}')
        print(f'{k} AP: {v[0]:.4f}', file=f)
    f.close()
    

def main(args):
    if args.dataset_name == 'kitti':
        val_dataset = Kitti(data_root=args.data_root, split='val')
        CLASSES = Kitti.CLASSES
    elif args.dataset_name == 'custom':
        val_dataset = Custom(data_root=args.data_root, split='val')
        CLASSES = Custom.CLASSES
    else: 
        raise ValueError("Dataset name should be 'kitti' or 'custom'")
    
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    ids_file = os.path.join(args.data_root, 'ImageSets', 'val.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]

    if not args.no_cuda:
        model = PointPillars(nclasses=args.nclasses).cuda()
        model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])
    else:
        model = PointPillars(nclasses=args.nclasses)
        model.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu'))['model_state_dict'])

    path_list = args.ckpt.split(os.sep)
    saved_path = os.path.join(args.saved_path, path_list[1])
    os.makedirs(saved_path, exist_ok=True)
    saved_submit_path = os.path.join(saved_path, 'submit')
    os.makedirs(saved_submit_path, exist_ok=True)

    pcd_limit_range = np.array([-1, -40, -3, 70.4, 40, 3], dtype=np.float32)           #prev value: [0, -40, -3, 70.4, 40, 0.0]
    index = 0

    model.eval()
    with torch.no_grad():
        format_results = {}
        print('Predicting and Formatting the results.')
        for i, data_dict in enumerate(tqdm(val_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batch_results = model(batched_pts=batched_pts, 
                                  mode='val',
                                  batched_gt_bboxes=batched_gt_bboxes, 
                                  batched_gt_labels=batched_labels)
            # pdb.set_trace()
            for j, result in enumerate(batch_results):
                format_result = {
                    'name': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                result_filter = keep_bbox_from_lidar_range_v2(result, pcd_limit_range)
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                for lidar_bbox, label, score in zip(lidar_bboxes, labels, scores):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['dimensions'].append(lidar_bbox[3:6])
                    format_result['location'].append(lidar_bbox[:3])
                    format_result['rotation_y'].append(lidar_bbox[6])
                    format_result['score'].append(score)

                index = (i * args.batch_size) + j
                idx = int(ids[index])
                write_label_filtered_with_score(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))

                format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
        
        write_pickle(format_results, os.path.join(saved_path, 'results.pkl'))
    
    print('Evaluating.. Please wait several seconds.')
    do_eval(format_results, val_dataset.data_infos, CLASSES, saved_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--saved_path', default='results', help='your saved path for predicted results')
    parser.add_argument('--dataset_name', default='custom')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nclasses', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    main(args)
