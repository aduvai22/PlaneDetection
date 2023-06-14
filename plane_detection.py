import cv2
import numpy as np
import open3d as o3d
from open3d import geometry
import matplotlib.pyplot as plt

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

#create paths and load data
folder = './bagfiles/data5/'
file = 'data5_0'
# rgb_folder = './bagfiles/rgb_folder/'
# rgb_file = 'rosbag2_2023_04_14-16_41_33'
# 'rosbag2_2023_04_12-17_57_22'
# 'rosbag2_2023_04_14-16_28_36'
# 'rosbag2_2023_04_14-16_41_33'
# 'rosbag2_2023_04_14-16_40_18'

# create reader instance and open for reading
with Reader(folder+file) as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/color/preview/image':
            img_msg = deserialize_cdr(rawdata, connection.msgtype)
        if connection.topic == '/stereo/depth':
            depth_msg = deserialize_cdr(rawdata, connection.msgtype)
#    

rgb = img_msg.data.reshape(250,250,3)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
# rgb = cv2.resize(rgb, (640,480), interpolation = cv2.INTER_AREA)
rgb_img = o3d.geometry.Image(rgb)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
gray_img = o3d.geometry.Image(gray)
plt.imshow(rgb)
# plt.show()
# depth_folder = './bagfiles/depth_folder/'
# depth_file = 'rosbag2_2023_04_14-16_41_48'
# 'rosbag2_2023_04_12-18_02_49'
# 'rosbag2_2023_04_14-16_28_49'
# 'rosbag2_2023_04_14-16_41_48'
# 'rosbag2_2023_04_14-16_40_29'

# create reader instance and open for reading
# with Reader(depth_folder+depth_file) as reader:
    # topic and msgtype information is available on .connections list
    # for connection in reader.connections:
    #     print(connection.topic, connection.msgtype)

    # iterate over messages
    # for connection, timestamp, rawdata in reader.messages():
#         if connection.topic == '/stereo/depth':
#             depth_msg = deserialize_cdr(rawdata, connection.msgtype)
#             print(depth_msg.header.frame_id)

f_depth = np.frombuffer(depth_msg.data, dtype=np.uint16)
f_depth = f_depth.reshape(480,640)
x,y,w,h = 195,115,250,250
f_depth = f_depth[y:y+h, x:x+w]
depth_img = o3d.geometry.Image(f_depth)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=True)

plt.subplot(1, 2, 1)
plt.title('Grayscale image')
plt.imshow(rgbd.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd.depth)
plt.show()

cam = o3d.camera.PinholeCameraIntrinsic()
cam.intrinsic_matrix =  [[989.7, 0.00, 320.1] , [0.00, 747.9, 281.1], [0.00, 0.00, 1.00]]
# cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=1000)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
# o3d.visualization.draw_geometries([pcd])
xyz = np.asarray(pcd.points)
colors = None
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
elif pcd.has_normals():
    colors = (0.5, 0.5, 0.5) + np.asarray(pcd.normals) * 0.5
else:
    geometry.paint_uniform_color((1.0, 0.0, 0.0))
    colors = np.asarray(geometry.colors)

segment_models={}
segments={}
max_plane_idx=20
rest=pcd
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(distance_threshold=1e-10,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")
    [a, b, c, d] = segment_models[i]  
    print(a,b,c,d)
    x = np.linspace(-10,10,100)
    y = np.linspace(-10,10,100)

    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c
    ax.view_init(45, -45)
    ax.plot_surface(X, Y, Z, alpha=0.75)
plt.show()