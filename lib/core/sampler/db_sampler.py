from lib.core.sampler.sampler_ops import DataBaseSampler
import pickle
import abc
import numpy as np

def DBSampler(config, logger=None):
    config_db_sampler = config.input.train.preprocess.db_sampler
    config_preprocess = config_db_sampler.db_preprocess_steps
    preprocess = []
    for step in list(config_preprocess.keys()):
        if step == "filter_by_min_num_points":
            prep = DBFilterByMinNumPoint(dict(zip(config_preprocess[step]["classes"],config_preprocess[step]["values"])), logger=logger)
        elif step == "filter_by_difficulty":
            prep = DBFilterByDifficulty(config_preprocess[step]["value"], logger=logger)
        else:
            raise ValueError("unknown database prep type")
        preprocess.append(prep)
    db_preprocess = DataBasePreprocessor(preprocess)
    rate = config_db_sampler.rate
    global_rotate_range = list(config_db_sampler.global_random_rotation_range_per_object)
    group_config = config_db_sampler.sample_groups
    group_classes = group_config.classes
    group_values = group_config.values
    groups = []
    for i in range(len(group_classes)):
        groups.append({group_classes[i]:group_values[i]})
   
    info_path = config_db_sampler.db_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    global_rotate_range = None if len(global_rotate_range) == 0 else global_rotate_range
    sampler = DataBaseSampler(db_infos, groups, db_preprocess, rate, global_rotate_range, logger=logger)
    return  sampler


class BatchSampler:
    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]

class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass


class DBFilterByDifficulty(DataBasePreprocessing):
    def __init__(self, removed_difficulties, logger=None):
        self._removed_difficulties = removed_difficulties
        logger.info("db_filter_by_removed_difficulties")

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict, logger=None):
        self._min_gt_point_dict = min_gt_point_dict
        logger.info("db_filter_by_min_num_points")

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


class DataBasePreprocessor:
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos

