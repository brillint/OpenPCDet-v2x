import argparse

import open3d


OPEN3D_FLAG = True

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import lidar_utils as ut

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:1.bin
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )




#
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]
# box_colormap = box_colormap.tolist()
box_colormap2 = np.random.random((50, 3)).tolist()





def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default='./cfgs/my_models/centerpoint.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', default='./detection_pth_lib/centerpoint/checkpoint_centerpoint_2100.pth', type=str,
                        help='specify the pretrained model')


    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')
    parser.add_argument('--camfusion', type=bool, default='True')
    parser.add_argument('--calib_json', type=str, default='./wanjiwviewpoint/0830/viewpoint_0830.json')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def gogo():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    vz = np.fromfile('aaa.bin')
    vz = vz.reshape(-1,4)

    vis,pts = ut.creat_vis()

    with torch.no_grad():


        input_dict = {
            'points': vz
        }
        data_dict = demo_dataset.prepare_data(input_dict)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
        # c_time = time.time()

        ref_boxes = pred_dicts[0]['pred_boxes']
        ref_scores = pred_dicts[0]['pred_scores']
        ref_labels = pred_dicts[0]['pred_labels']

        ref_boxes_car = ref_boxes[(ref_scores > 0.2) & (ref_labels == 1)]
        ref_boxes_car_label = ref_labels[(ref_scores > 0.2) & (ref_labels == 1)]

        ref_boxes_person = (ref_boxes[(ref_scores > 0.3) & (ref_labels == 2)])
        ref_boxes_person_label = ref_labels[(ref_scores > 0.3) & (ref_labels == 2)]

        ref_boxes_truck = (ref_boxes[(ref_scores > 0.3) & (ref_labels == 3)])
        ref_boxes_truck_label = ref_labels[(ref_scores > 0.3) & (ref_labels == 3)]

        ref_boxes_ride = (ref_boxes[(ref_scores > 0.5) & (ref_labels == 4)])
        ref_boxes_ride_label = ref_labels[(ref_scores > 0.5) & (ref_labels == 4)]

        ref_boxes_bus = (ref_boxes[(ref_scores > 0.3) & (ref_labels == 5)])
        ref_boxes_bus_label = ref_labels[(ref_scores > 0.3) & (ref_labels == 5)]

        ref_boxes_Smallbus = (ref_boxes[(ref_scores > 0.0) & (ref_labels == 6)])
        ref_boxes_Smallbus_label = ref_labels[(ref_scores > 0.0) & (ref_labels == 6)]

        ref_boxes_Engineeringtruck = (ref_boxes[(ref_scores > 0.0) & (ref_labels == 7)])
        ref_boxes_Engineeringtruck_label = ref_labels[(ref_scores > 0.0) & (ref_labels == 7)]
        ref_boxes_Trimotorcycle = (ref_boxes[(ref_scores > 0.3) & (ref_labels == 8)])
        ref_boxes_Trimotorcycle_label = ref_labels[(ref_scores > 0.3) & (ref_labels == 8)]

        ref_boxes_obj = torch.cat(
            [ref_boxes_car, ref_boxes_person, ref_boxes_truck, ref_boxes_ride, ref_boxes_bus, ref_boxes_Smallbus,
             ref_boxes_Engineeringtruck, ref_boxes_Trimotorcycle], dim=0)
        ref_boexes_obj_label = torch.cat(
            [ref_boxes_car_label, ref_boxes_person_label, ref_boxes_truck_label, ref_boxes_ride_label,
             ref_boxes_bus_label, ref_boxes_Smallbus_label, ref_boxes_Engineeringtruck_label,
             ref_boxes_Trimotorcycle_label], dim=0)
        ref_boxes_obj = ref_boxes_obj.cpu().numpy()
        # print(ref_boxes_car)

        ref_boxes_car = ref_boxes_car.cpu().numpy()
        # ref_boxes_person = ref_boxes.cpu() .numpy()

        bb_liv = []

        # hwlxyzr_car, _ = box2hwlxyzr(ref_boxes_car, bb_liv)
        # hwlxyzr_person, _ = box2hwlxyzr(ref_boxes_person, bb_liv)

        box3ds = ref_boxes_obj

        label_map = {1: 1, 2: 4, 3: 2, 4: 5, 5: 3}  #


        pts.points = open3d.utility.Vector3dVector(vz[:, :3])
        pts.colors = open3d.utility.Vector3dVector(np.ones((vz.shape[0], 3)))
        vis = ut.draw_box3(vis, box3ds, (0, 1, 0), ref_boexes_obj_label, ref_scores)
        vis.add_geometry(pts)

        # param = open3d.io.read_pinhole_camera_parameters(
        #     './wanjiwviewpoint/viewpoint0808.json')
        # dir ='./wanjiwviewpoint/0921/viewpoint_0921.json'

        param = open3d.io.read_pinhole_camera_parameters(
            'viewpoint1008.json')
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.run()
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
        zk = 1
        batch = 1







if __name__ == '__main__':

    gogo()

