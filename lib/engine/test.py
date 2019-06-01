from tqdm import tqdm
import torch
from lib.engine.convert_batch_to_device import convert_batch_to_device
from lib.utils.dist_common import synchronize, is_main_process
from lib.utils.dist_common import get_world_size, all_gather
from lib.utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = []
    cpu_device = torch.device("cpu")
    results_dict = {}
    for batch in tqdm(data_loader):
        example = convert_batch_to_device(batch, device=device)
        with torch.no_grad():
            outputs = model(example)
            for output in outputs:
                token = output['metadata']['token']
                for k, v in output.items():
                    if k not in ['metadata',]:
                        output[k] = v.to(cpu_device)
                results_dict.update({token: output,})
    model.train()
    return results_dict


def accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions



def test(dataloader, model, save_dir=None, device='cuda', distributed=False, logger=None):
    if distributed:
        model = model.module
   
    dataset = dataloader.dataset
    device = torch.device(device)
    num_devices = get_world_size()
    logger.info("Start evaluation on dataset for {} samples.".format(len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    
    detections = compute_on_dataset(model, dataloader, device)

    synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset),
            num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".
        format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        ))



    predictions = accumulate_predictions_from_multiple_gpus(detections)

    if not is_main_process():
        return
    result_dict = dataset.evaluation(predictions, save_dir)
    for k, v in result_dict["results"].items():
        logger.info(f"Evaluation {k}: {v}")
