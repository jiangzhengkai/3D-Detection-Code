from torch.utils.data import Dataset



class DatasetWarpper(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset


def build_dataset(config, training, voxel_generator, target_assiginer, dist):
    num_points_features = config.MODEL.NUM_POINTS.FEATURES


def build_dataloader(config, training, voxel_generator, target_assigner, dist):
    dataset = build_dataset(config, training, voxel_generator, target_assigner, dist)
    dataset = DatasetWarpper(dataset)

    batch_size = config.TRAIN.batch_size if training else config.TEST.batch_size
    
    if dist:
    
    else:
        num

    dataloader = torch.utils.data.Dataloader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             worker_init_fn=)



    return dataloader
