import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
PATH = "./data/0000000021.pcd"
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(PATH)
# print how many points there are
print(pcd)



###############
## Filtering ##
###############

# Downsampling: e.g converting points to voxels
print("Downsample the point cloud with a voxel of 0.1")
downpcd = pcd.voxel_down_sample(voxel_size=0.1)
# o3d.visualization.draw_geometries([downpcd])


# Remove outliers
print("Statistical oulier removal")
cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
cropping_inlier_cloud = downpcd.select_by_index(ind)
cropping_outlier_cloud = downpcd.select_by_index(ind, invert=True)
cropping_outlier_cloud.paint_uniform_color([0, 1, 0])
cropping_inlier_cloud.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([cropping_inlier_cloud, cropping_outlier_cloud])






##################
## Segmentation ##
##################

# Segmentation (the inlier is the ground, outlier is everything else)
plane_model, inliers = cropping_inlier_cloud.segment_plane(distance_threshold=0.2,
                                         ransac_n=30,
                                         num_iterations=800)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
segmentation_inlier_cloud = cropping_inlier_cloud.select_by_index(inliers)
segmentation_inlier_cloud.paint_uniform_color([0, 1.0, 0])
segmentation_outlier_cloud = cropping_inlier_cloud.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([segmentation_inlier_cloud, segmentation_outlier_cloud])
# o3d.visualization.draw_geometries([segmentation_outlier_cloud])







################
## Clustering ##
################

# DBSCAN clustering
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(    
        segmentation_outlier_cloud.cluster_dbscan(eps=0.5, min_points=20, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
segmentation_outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([segmentation_outlier_cloud])


