import numpy as np
import pickle
import json
import sklearn.metrics

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
#from lib.datasets.nuscenes_dataset import _get_available_scenes, _fill_trainval_infos
nusc = NuScenes(version='v1.0-trainval', dataroot="/data/datasets/NuScenes", verbose=True)
'''
val_scenes = splits.val
train_scenes = splits.train
available_scenes = _get_available_scenes(nusc)
available_scene_names = [s["name"] for s in available_scenes]

val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])


_, val_nusc_infos = _fill_trainval_infos(nusc,
                                         train_scenes,
                                         val_scenes,
                                         False,
                                         nsweeps=10)
'''
val_nusc_infos = pickle.load(open("/data/datasets/NuScenes/infos_val_10sweeps_withvelo.pkl",'rb'))

detection_results = json.load(open("results_63.42.json",'r'))

masks = {}
i = 0
for val_nusc_info in val_nusc_infos:
    mask_info = {}
    sample_token = val_nusc_info['token']

    #if sample_token == 'b4ff30109dd14c89b24789dc5713cf8c':
    #    import pdb;pdb.set_trace()
    scene = nusc.get('scene', nusc.get('sample', sample_token)['scene_token'])
    scene_token = scene['token']
    # Get records from the database.
    scene_record = nusc.get('scene', scene_token)
    log_record = nusc.get('log', scene_record['log_token'])
    map_record = nusc.get('map', log_record['map_token'])
    map_mask = map_record['mask']

    sample_record = nusc.get('sample', sample_token)
    # Poses are associated with the sample_data. Here we use the lidar sample_data.
    sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

    map_pose = np.concatenate(map_mask.to_pixel_coords(pose_record['translation'][0], pose_record['translation'][1]))

    xmin = map_pose[0] - 50.4 / map_mask.resolution
    ymin = map_pose[1] - 50.4 / map_mask.resolution

    xmax = map_pose[0] + 50.4 / map_mask.resolution
    ymax = map_pose[1] + 50.4 / map_mask.resolution

    mask = map_mask.mask()[int(ymin):int(ymax), int(xmin):int(xmax)]
    det_coors_on_map = []
    dets = detection_results["results"][sample_token]
    for det in dets:
        #det_coor = np.concatenate(map_mask.to_pixel_coords(det['translation'][0], det['translation'][1]))
        det_coor = map_mask.is_on_mask(det['translation'][0], det['translation'][1],dilation=0)
        print('is_on_mask',det_coor)
        det_coors_on_map.append(det_coor)
    mask_info["mask"] = mask
    mask_info["coors"] = [int(xmin), int(ymin)]
    mask_info["map_shape"] = map_mask.mask().shape
    mask_info["resolution"] = map_mask.resolution
    mask_info["det_coors_on_map"] = det_coors_on_map
    masks[sample_token] = mask_info
    print(i)
    i = i + 1

#res_path = "mask_infos.pkl"
#with open(res_path,'wb') as f:
#    pickle.dump(masks, f)
#print("finish generat mask infos")

mask_infos = masks
import json
#import pickle
#mask_infos = pickle.load(open('mask_infos.pkl','rb'))
detection_results = json.load(open("results_63.42.json",'r'))

new_detections = {}
sample_tokens = detection_results["results"].keys()
i = 0
for sample_token in sample_tokens:
    detection = detection_results["results"][sample_token]
    new_detection = []
    mask_info = mask_infos[sample_token]
    #import pdb;pdb.set_trace()
    for j in range(len(detection)):
        flag = mask_info['det_coors_on_map'][j]
        #x = x - mask_info['coors'][0]
        #y = y - mask_info['coors'][1]
        #if 0< int(x) < 1008 and 0<int(y) < 1008 and mask_info['mask'][int(y-3):int(y+2), int(x-3):int(x+2)].sum()>0:
        now_detection = {}
        if flag:
            now_detection.update(detection[j])
            now_detection["on_road"] = True
        else:
            now_detection.update(detection[j])
            now_detection["on_road"] = True


        new_detection.append(now_detection)
    i = i + 1
    print(i)
    new_detections[sample_token] = new_detection

new_detections_results = {}
new_detections_results['results'] = new_detections
new_detections_results['meta'] = detection_results['meta']

with open("results_63.42_new.json",'w') as f:
    json.dump(new_detections_results,f)
