from lib.core.datasets.preprocess import DBFilterByDifficulty, DBFilterByMinNumPoint, DataBasePreprocessor
from lib.core.database.sampler_ops import DataBaseSampler
import pickle

def DBSampler(config):
    config_preprocess = config.input.train.preprocess.db_sampler

    preprocess = []
    for step in list(config_preprocess.keys()):
        if step == "filter_by_min_num_points":
            prep = DBFilterByMinNumPoint(dict(zip(config_preprocess[step]["classes"],config_preprocess[step]["values"])))
        elif step == "filter_by_difficulty"
            prep = DBFilterByDifficulty(config_preprocess[step]["value"])
        else:
            raise ValueError("unknown database prep type")
        preprocess.append(prep)
    
    db_preprocess = DataBasePreprocessor(preprocess)
    rate = config_preprocess.rate
    global_rotate_range = list(config_preprocess.global_random_rotation_range_per_object)
    groups = config_preprocess.sample_groups
    info_path = config_preprocess.db_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    global_rotate_range = None if len(global_rotate_range) == 0
    sampler = DataBaseSampler(db_infos, groups, db_preprocess, rate, global_rotate_range)
    return  sampler
