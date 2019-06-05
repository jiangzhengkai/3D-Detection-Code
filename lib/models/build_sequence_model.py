from lib.core.voxel.voxel_generator import VoxelGenerator
from lib.core.target.target_assigner import target_assigners_all_classes
from lib.models.voxelnetsequence import VoxelNetSequence


def build_sequence_network(config, logger=None, device=None):
    ######## voxel generator ########
    voxel_generator = VoxelGenerator(
        voxel_size=config.input.voxel.voxel_size,
        point_cloud_range=config.input.voxel.point_cloud_range,
        max_num_points=config.input.voxel.max_num_points)
    
    ######## target assiginers ########
    target_assigners = target_assigners_all_classes(config)

    vfe_num_filters = config.model.encoder.vfe.num_filters
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_classes = [len(target_assigner.classes) for target_assigner in target_assigners]


    net = VoxelNetSequence(
            output_shape=dense_shape,
            config=config,
            num_classes=num_classes,
            use_norm=True,
            target_assigners=target_assigners,
            voxel_generator=voxel_generator,
            name='VoxelNetSequence',
            device=device,
            logger=logger)
    return net    

