from torch.utils.data import Dataset


voxel_generator
target_assigners


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


def build_dataset(config, training):

    voxel_generator = build_voxel_generator(config)
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // 2
    




def build_dataloader(config, training):
    dataset = build_dataset(config, training)
    dataset = DatasetWarpper(dataset)
    
    batch_size = config.train.batch_size if training else config.eval.batch_size
    num_workers = config.train.num_workers if training else config.eval.num_workers
    sampler = train_sampler if training else eval_sampler

    dataloader = torch.utils.data.Dataloader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_batch_fn,
					     sampler=sampler)
    return dataloader
