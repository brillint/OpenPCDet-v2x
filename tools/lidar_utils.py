import open3d
OPEN3D_FLAG = True
import numpy as np



box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 2, 1],
    [1, 3, 0],
[1, 4, 0],
[1, 5, 0],
[1, 6, 0]
]
box_colormap2 = np.random.random((50, 3)).tolist()

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box3(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    ref_labels = ref_labels.tolist()
    gt_boxes = np.array(gt_boxes)
    for i in range(gt_boxes.shape[0]):

        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            pp = ref_labels[i]
            vo = box_colormap[pp]
            line_set.paint_uniform_color(vo)
            # print(ref_labels[i])
        vis.add_geometry(line_set)

    return vis

def creat_vis():
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    pts = open3d.geometry.PointCloud()

    return vis,pts

