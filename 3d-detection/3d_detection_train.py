#### python package
import pathlib
import torch
from 3d-detection.config import cfg









### main function
def train(config, model_dir, result_path=None, resume=False):
    ####### load config #######

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ####### build network #######
    net = build_network(config).to(device)
    
    


 
    ####### dataloader #######
    dataloader = build_dataloader(config, training=True, voxel_generator, target_assigner)



    ####### optimizer #######
    optimizer = build_optimizer(config)
    lr_scheduler = build_scheduler(config)



    ####### training #######
    data_iter = iter(dataloader)
    


def evaluate(config, model_dir, result_path=None, model_path=None, measure_time=False, batch_size=None):

    ####### load config #######



    ####### build network #######



    ####### dataloader #######



    ####### evaluation #######
