from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint_rcnn import CenterPointRCNN
from .centerpoint import CenterPoint
from .pillarnet import PillarNet
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'CenterPointRCNN': CenterPointRCNN,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet
}


def build_detector(model_cfg, num_class, dataset):
    """
    根据配置文件中的网络名称，构造对应的模型
    """
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
