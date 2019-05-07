import torch
import pickle






class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, info_path, class_names=None, prep_func=None, num_point_features=None, **kwargs):
        assert info_path is not None
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = 4
        self._class_name = class_name
        self._prep_func = prep_func

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._kitti_infos[0]
            return None
        gt_annos = [info["annos"] for info in self._kitti_infos]
        return gt_annos

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["image"]["image_idx"]) for info in self._kitti_infos]

        annos = []
        for det_idx in gt_image_idxes:
            det = detection[det_idx]
            info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
    



