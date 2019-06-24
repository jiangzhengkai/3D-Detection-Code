import numpy as np

def get_downsample_factor(config):

    decoder_config = config.model.decoder
    downsample_factor = np.prod(decoder_config.rpn.downsample_layer_strides)

    if len(decoder_config.rpn.upsample_layer_strides) > 0:
        downsample_factor /= decoder_config.rpn.upsample_layer_strides[-1]

    if "middle" in config.model.encoder:
        downsample_factor *= config.model.encoder.middle.downsample_factor

    assert downsample_factor > 0
    return downsample_factor
