import json
import numpy as np
names_all = ["car", "construction_vehicle", "bus", "truck", "trailer"]

detections = json.load(open("results_63.42_new.json",'r'))

new_detections = {}

keys = list(detections["results"].keys())
for det_key in keys:
    new_dets = []
    dets = detections["results"][det_key]
    for det in dets:
        import pdb;pdb.set_trace()
        new_det = det
        velocity = det["velocity"]
        detection_name = det["detection_name"]
        attribute_name = det["attribute_name"]
        on_road = det["on_road"]
        velo = np.sqrt(velocity[0]**2+velocity[1]**2)
        if detection_name in ["bus"]:
            if on_road and velo < 0.1:
                new_det["attribute_name"] = 'vehicle.stopped'
        elif detection_name in ["car","truck","trailer"]:
            if on_road and velo > 0.1:
                new_det["attribute_name"] = 'vehicle.moving'
            elif velo <= 0.1:
                new_det["attribute_name"] = 'vehicle.parked'
        elif detection_name in ["bicycle", "motorcycle"]:
            if on_road and velo > 0.1:
                new_det["attribute_name"] = 'cycle.with_rider'
            elif np.sqrt(velocity[0]**2 +velocity[1]**2) <= 0.1:
                new_det["attribute_name"] = 'cycle.without_rider'
        elif detection_name in ["pedestrain"]:
            if np.sqrt(velocity[0]**2 +velocity[1]**2) >= 0.1:
                attr = 'pedestrian.moving'
            elif not on_road and np.sqrt(velocity[0]**2 +velocity[1]**2) < 0.1:
                attr = 'pedestrain.sitting_lying_down'
            elif on_road and np.sqrt(velocity[0]**2 +velocity[1]**2) < 0.1:
                attr = 'pedestrian.standing'
        new_dets.append(new_det)
    new_detections[det_key] = new_dets


new_detections_results = {}
new_detections_results['results'] = new_detections
new_detections_results['meta'] = detections['meta']
with open("results_63.42_hand.json",'w') as f:                                                                                             json.dump(new_detections_results,f)
