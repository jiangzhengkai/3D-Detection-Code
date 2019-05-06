import os
import pathlib
import numpy as np
from skimage import io
import concurrent.features as features


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)

def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)

    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
 
    if not (prefix / file_path):
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def get_image_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training, relative_path)

def get_label_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training, relative_path)

def get_velodyne_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training, relative_path)

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True)
    """
    KITTI annotation format {
        points: [N, 3+] point cloud
        image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annotations: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            difficulty: kitti difficulty
            group_ids: used for muti-part object
        }
    }
    """
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_function(idx):
        info = {}
        point_cloud_info = {'num_features': 4}
        calib_info = {}
        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            point_cloud_info['velodyne_path'] = get_velodyne_path(idx, path, training, relative_path)
        image_info['image_path'] = get_image_path(idx, path, training, relative_path)

        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
            
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_annotation(label_path)
        info['image'] = image_info
        info['point_cloud'] = point_cloud_info
        if calib:
            calib_path = get_calib_path(idx, path, training, relative_path=False)
            with open(calib_path,'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect            
            Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info["calib"] = calib_info
            
        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info
    with features.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)

def add_difficulty_to_annos(info):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff



def get_label_annotation(label_path):
    annotations = {}
    annotations.update({'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimentions': [],
                        'location': [],
                        'rotation_y': []})
    with open(label_path, 'r') as f:
        lines = f.readlines()

    context = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw format
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def get_label_annotations(label_folder, image_ids=None):
    if image_ids is None:
        file_paths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + '.txt')
        anno = get_label_anno(label_filename)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos 
