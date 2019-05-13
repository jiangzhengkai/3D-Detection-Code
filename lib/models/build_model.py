from lib.core.voxel.voxel_generator import VoxelGenerator


def build_network(config, logger=None, device=None):
    ######## voxel generator ########
    voxel_generator = VoxelGenerator(
        voxel_size=config.input.voxel.voxel_size,
        point_cloud_range=config.input.voxel.point_cloud_range,
        max_num_points=config.input.voxel.max_num_points,
        max_voxels=config.input.voxel.max_voxels)
    
    ######## target assiginers ########
    target_assigners = []
    class_names = []
    flag = 0
    for num_class, class_name in zip(config.tasks.num_classes, config.tasks.class_names):
        target_assigner = TargetAssigner(
            box_coder=bbox_coder,
            anchor_generators=anchor_generators_all_class[flag:flag+len(class_name)],
            region_similarity_calculator=region_similarity,
            positive_fraction=config.target_assigner.anchor_generators.sample_positive_fraction,
            sample_size=config.target_assigner.anchor_generators.sample_size)

        flag += len(class_name)
        target_assigners.append(target_assigner)
        class_names.append(class_name)

    vfe_num_filters = config.model.encoder.vfe.num_filters
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    num_classes = [len(target_assigner.classes) for target_assigner in target_assigners]


    net = VoxelNet(
            dense_shape,
            num_classes=num_classes,
            use_norm=True,
            config,
            target_assigners=target_asigners,
            voxel_generator=voxel_generator,
            device=device,
            logger=logger)
    return net    

