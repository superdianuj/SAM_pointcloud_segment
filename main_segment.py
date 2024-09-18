import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import gc
import os
import requests
import cv2
import time

def get_model():
    if not os.path.exists('model'):
        os.mkdir('model')
        url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        local_filename = 'model/sam.pth'

        if os.path.exists(local_filename):
            print(f'File {local_filename} already exists. Skipping download.')
        else:
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with open(local_filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)

                print(f'File downloaded successfully and saved as {local_filename}')
            else:
                print(f'Failed to download file. HTTP Status Code: {response.status_code}')
    else:
        print('SAM already exists and will be loaded')


def sam_masks(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    c_mask = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.8)))
        c_mask.append(img)
    return c_mask


def cloud_to_image(pcd_np, resolution):
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    print(f"Point Cloud Bounds: X[{minx}, {maxx}], Y[{miny}, {maxy}]")
    
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    print(f"Computed Image Dimensions: Width={width}, Height={height}")
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        
        # Ensure the pixel coordinates are within image bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            image[pixel_y, pixel_x] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return image


def read_data(file_path):
    if file_path.endswith('.ptx'):
        with open(file_path, 'r') as file:
            for _ in range(12):
                file.readline()

            points = []
            colors = []

            for line in file:
                parts = line.strip().split()
                if len(parts) == 7:
                    x, y, z, intensity, r, g, b = map(float, parts)
                    points.append([x, y, z])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])

        points, colors = np.array(points), np.array(colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_np = np.concatenate((points, colors), axis=1)

    elif file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd_np = np.concatenate((points, colors), axis=1)
    else:
        raise ValueError('Unsupported file format')
    return pcd_np


def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):
    # Translate the point cloud by the negation of the center coordinates
    translated_points = point_cloud - center_coordinates

    # Convert 3D point cloud to spherical coordinates
    r = np.linalg.norm(translated_points, axis=1)
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
    phi = np.arccos(translated_points[:, 2] / r)

    # Map spherical coordinates to pixel coordinates
    resolution_x = 2 * resolution_y
    x = ((theta + np.pi) / (2 * np.pi) * resolution_x).astype(int)
    y = ((phi / np.pi) * resolution_y).astype(int)

    # Create the spherical image with RGB channels
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    # Create the mapping between point cloud and image coordinates
    mapping = np.full((resolution_y, resolution_x), -1, dtype=int)

    # Assign points to the image pixels
    for i in range(len(translated_points)):
        ix = np.clip(x[i], 0, resolution_x - 1)
        iy = np.clip(y[i], 0, resolution_y - 1)
        if mapping[iy, ix] == -1 or r[i] < r[mapping[iy, ix]]:
            mapping[iy, ix] = i
            image[iy, ix] = (colors[i] * 255).astype(np.uint8)
    
    return image, mapping

def color_point_cloud(image_path, point_cloud, mapping):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    modified_point_cloud = np.zeros((point_cloud.shape[0], point_cloud.shape[1]+3), dtype=np.float32)
    modified_point_cloud[:, :3] = point_cloud
    for iy in range(h):
        for ix in range(w):
            point_index = mapping[iy, ix]
            if point_index != -1:
                color = image[iy, ix]/255.0
                modified_point_cloud[point_index, 3:] = color
    return modified_point_cloud



def main():
    get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam = sam_model_registry["vit_h"](checkpoint='model/sam.pth')
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam.to(device)

    # Get memory allocated and reserved in bytes
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()

    # Convert bytes to gigabytes
    allocated_gb = allocated_bytes / (1024**3)
    reserved_gb = reserved_bytes / (1024**3)

    print('Mem allocated by other programs: {:.2f} GB, reserved: {:.2f} GB'.format(allocated_gb, reserved_gb))

    gc.collect()
    torch.cuda.empty_cache()
    
    pcd_np = read_data('scan0.ptx')
    
    resolution = 0.05  # Adjust resolution for better output
    orthoimage = cloud_to_image(pcd_np, resolution)
    print("Ortho-Image Shape: ", orthoimage.shape)

    dpi = 100  # Dots per inch
    height, width, _ = orthoimage.shape
    figsize = width / dpi, height / dpi
    fig = plt.figure(figsize=figsize)
    fig.add_axes([0, 0, 1, 1])
    plt.imshow(orthoimage)
    plt.axis('off')
    plt.savefig("orthoimage.jpg", dpi=dpi, bbox_inches='tight', pad_inches=0)
    print("Ortho-Image saved as orthoimage.jpg")

    resolution = 500

    #Defining the position in the point cloud to generate a panorama
    center_coordinates = np.mean(pcd_np[:, :3], axis=0)

    points=pcd_np[:, :3]
    colors=pcd_np[:, 3:]

    #Function Execution
    spherical_image, mapping= generate_spherical_image(center_coordinates, points, colors, resolution)
    print("Spherical Projection Shape: ", spherical_image.shape)
    dpi=100
    #Plotting with matplotlib
    fig = plt.figure(figsize=(np.shape(spherical_image)[1]/dpi, np.shape(spherical_image)[0]/dpi))
    fig.add_axes([0,0,1,1])
    plt.imshow(spherical_image)
    plt.axis('off')

    #Saving to the disk
    plt.savefig("spherical_projection.jpg")
    print("Spherical Projection saved as spherical_projection.jpg")


    temp_img = cv2.imread("spherical_projection.jpg")
    image_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    result = mask_generator.generate(image_rgb)
    t1 = time.time()
    print(f"Time taken by SAM segmentation: {t1 - t0:.2f} seconds")

    dpi=100

    fig = plt.figure(figsize=(np.shape(image_rgb)[1]/dpi, np.shape(image_rgb)[0]/dpi))
    fig.add_axes([0,0,1,1])

    plt.imshow(image_rgb)
    color_mask = sam_masks(result)
    plt.axis('off')
    plt.savefig("spherical_projection_segmented.jpg")
    print("Segmented Spherical Projection saved as spherical_projection_segmented.jpg")

    modified_point_cloud = color_point_cloud("spherical_projection_segmented.jpg", points, mapping)

    # Save the segmented point cloud as a .ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(modified_point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(modified_point_cloud[:, 3:])
    o3d.io.write_point_cloud("segmented_point_cloud.ply", pcd)
    print("Segmented Point Cloud saved as segmented_point_cloud.ply")



if __name__ == '__main__':
    main()
