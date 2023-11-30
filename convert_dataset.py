import argparse
from csv import reader
import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import os

from utils import write_points, write_label_no_score


def truncate(rows):
    result = []
    for row in rows:
        if row.find('.') != -1:
            index = row.find('.') + 3
            result.append(row[:index])
        else:
            result.append(row + '.00')

    return result

def filter_files(files, suffix):
    result = []
    for file in files:
        if file.endswith(suffix):
            result.append(file)
    return result


def read_bag(bag_root, bag_dir, pc_topic='/mec_algo_cloud'):
    msgs = []
    bag = rosbag.Bag(os.path.join(bag_root, bag_dir))
    for topic, msg, t in bag.read_messages(topics=[pc_topic]):
        msgs.append(msg)
    bag.close()
    return msgs


def read_csv(csv_root, csv_dir):
    rows = {}
    suffix = os.path.splitext(csv_dir)[1]
    assert suffix == '.csv'
    with open(os.path.join(csv_root, csv_dir)) as tt:
        csvreader = reader(tt)
        header = next(csvreader)
        row_id = 0
        for row in csvreader:
            if row[1]:
                rows[row_id] = row
            row_id += 1
    tt.close()
    return rows


def convert_bag2bin(msg, output_root, id):
    pc = point_cloud2.read_points_list(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
    pc = np.array(pc, dtype=np.float32)
    out_filename = '{:0>6}.bin'.format(str(id))
    output_file_name = '{0}/{1}'.format(output_root, out_filename)
    write_points(pc, output_file_name)


def convert_csv2txt(row, output_root, id):
    out_filename = '{:0>6}.txt'.format(str(id))
    output_file_name = '{0}/{1}'.format(output_root, out_filename)
    result = {
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    }
    result['name'].append('Wheelchair')
    result['truncated'].append('0.00')
    result['occluded'].append('3')
    result['alpha'].append('-10')
    result['bbox'].append(np.array(['0.00' for _ in range(4)]))
    result['dimensions'].append(truncate(row[4:7]))
    result['location'].append(truncate(row[1:4]))
    result['rotation_y'].append(truncate([row[9]]))

    write_label_no_score(result, output_file_name)


def convert_dataset(bag_root, csv_root, pc_output_root, label_output_root, start_id):
    if not os.path.exists(bag_root) or not os.path.exists(csv_root):
        raise FileNotFoundError

    if pc_output_root is None:
        pc_output_root = os.path.join(os.path.abspath(bag_root), 'velodyne')

    if label_output_root is None:
        label_output_root = os.path.join(os.path.abspath(csv_root), 'label')

    os.makedirs(pc_output_root, exist_ok=True)
    os.makedirs(label_output_root, exist_ok=True)
    csv_files = filter_files(os.listdir(csv_root), '.csv')
    bag_files = filter_files(os.listdir(bag_root), '.bag')
    assert len(csv_files) == len(bag_files)

    for file_id in range(len(csv_files)):
        assert os.path.splitext(csv_files[file_id])[0] == os.path.splitext(bag_files[file_id])[0]
        msgs = read_bag(bag_root, bag_files[file_id])
        rows = read_csv(csv_root, csv_files[file_id])
        print(f"Converting {bag_files[file_id]} and {csv_files[file_id]}")
        for row in rows:
            convert_bag2bin(msgs[row], pc_output_root, start_id)
            convert_csv2txt(rows[row], label_output_root, start_id)
            start_id += 1


def main(args):
    bag_root = args.bag_data_root
    csv_root = args.csv_data_root
    pc_output_root = args.pc_output_root
    label_output_root = args.label_output_root
    start_id = args.start_id

    convert_dataset(bag_root, csv_root, pc_output_root, label_output_root, start_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--bag_data_root', default='../datasets/ROSBAG',
                        help='your root for rosbag datas', required=True)
    parser.add_argument('--csv_data_root', default='../datasets/ROSBAG/export',
                        help='your root for csv files', required=True)
    parser.add_argument('--pc_output_root', help='output root for point cloud data')
    parser.add_argument('--label_output_root', help='output root for label data')
    parser.add_argument('--start_id', help='starting id for output files', default=0, type=int)
    args = parser.parse_args()

    main(args)
