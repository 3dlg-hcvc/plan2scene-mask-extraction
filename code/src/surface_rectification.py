import numpy as np
import math
from PIL import Image

def get_origin(x,y,depth_map, vertical_fov = np.pi/4, horizontal_fov = np.pi/3):
    x_mid = depth_map.shape[0]/2
    x_delta = -(x - x_mid)
    x_angle = (horizontal_fov/2) * (x_delta/x_mid)
    y_mid = depth_map.shape[1] / 2
    y_delta = -(y - y_mid)
    y_angle = (vertical_fov / 2) * (y_delta / y_mid)
    depth = depth_map[y,x]
    dx = np.tan(x_angle)*depth
    dy = np.tan(y_angle)*depth
    point = (dx,dy,depth)
    return point

def get_vertices(img_interested):
    verts_pq_orig = []
    np_img_interested = np.array(img_interested)
    for y in range(0, np_img_interested.shape[0]):
        for x in range(0, np_img_interested.shape[1]):
            if np_img_interested[y,x,3] > 0:
                verts_pq_orig.append((float(x)/np_img_interested.shape[1], float(y)/np_img_interested.shape[0]))
    return verts_pq_orig



def get_tangents(normal):
    up = np.array([0,1,0])
    d1 = np.cross(up,normal)
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.cross(d1,normal)
    d2 = d2 / np.linalg.norm(d2)
    return d1,d2


def get_largest_inscribed_square(mask):
    """
    Largest Inscribed Square
    """
    # This is a brute-force approximation implementation. 
    # A better solution exists using dynamic programming.
    largest = 0
    step_size= 5
    delta_size = 10
    largest_conf = -1,-1,-1,-1
    for x_min in range(0, mask.shape[1],step_size):
        for y_min in range(0, mask.shape[0],step_size):
            for length in range(20, min(mask.shape[1]-x_min, mask.shape[0]-y_min),delta_size):
                x_max = x_min + length
                y_max = y_min + length
                sub_mask = mask[y_min:y_max, x_min:x_max]
                if (y_max-y_min)*(x_max-x_min) > largest:
                    if (sub_mask==False).sum() == 0:
                        largest = (y_max-y_min)*(x_max-x_min)
                        largest_conf = x_min,y_min, x_max,y_max
    return largest_conf

def get_rectified_mask(img_interested, colours_np, depth_np, instance_param_np, param_np, predict_segmentation_np , inscribed_rect, vertical_fov = np.pi/4, horizontal_fov = np.pi/3):
    central_point = ((inscribed_rect[0] + inscribed_rect[2])/2,(inscribed_rect[1] + inscribed_rect[3])/2)
    central_point = (int(central_point[0]), int(central_point[1]))
    
    central_point_plane = (central_point[0]/img_interested.size[0]*colours_np.shape[1],central_point[1]/img_interested.size[1]*colours_np.shape[0])
    central_point_plane = tuple([int(a) for a in central_point_plane])
    
    plane_id = predict_segmentation_np[central_point_plane[1],central_point_plane[0]]
    if plane_id != 20: # No plane
        normal = instance_param_np[:,plane_id]
    else:
        normal = param_np[0,:,central_point_plane[1],central_point_plane[0]]
  
    normal[0] = -normal[0]
    d1,d2 = get_tangents(normal)
    
    point = get_origin(central_point_plane[0],central_point_plane[1],depth_np,vertical_fov, horizontal_fov)
    
    normal_matrix =[d1[0],d1[1],d1[2],0, d2[0],d2[1],d2[2],0, normal[0],normal[1],normal[2],0,point[0],point[1], -point[2],1]
    
    verts_pq_orig = get_vertices(img_interested)
    
    rectified, verts_ij = rectify(img_interested, normal_matrix, verts_pq_orig, fov=60)
    return rectified, verts_ij


# Code adapted from OpenSurfaces download script

# We use rectification implementation by OpenSurfaces: http://opensurfaces.cs.cornell.edu/publications/opensurfaces/#download.
# Refer the helper script at the above link.


def projection_function(homography):
    """
    Returns a function that applies a homography (3x3 matrix) to 2D tuples
    """
    H = np.copy(homography)

    def project(uv):
        xy = H * np.matrix([[uv[0]], [uv[1]], [1]])
        return (float(xy[0] / xy[2]), float(xy[1] / xy[2]))
    return project

def transform(H, points):
    proj = projection_function(H)
    return [proj(p) for p in points]

def bbox_vertices(vertices):
    """
    Return bounding box of this object, i.e. ``(min x, min y, max x, max y)``

    :param vertices: List ``[[x1, y1], [x2, y2]]`` or string
        ``"x1,y1,x2,y2,...,xn,yn"``
    """


    x, y = zip(*vertices)  # convert to two lists
    return (min(x), min(y), max(x), max(y))


def rectify(photo_image, normal_matrix,verts_pq_orig, fov=60):
    w, h = photo_image.size
    focal_pixels = 0.5 * max(w, h) / math.tan(math.radians(0.5 * fov))
    # uvnb: [u v n b] matrix arranged in column-major order
    uvnb = [float(f) for f in normal_matrix]
    verts_pq = [(v[0] * w, v[1] * h) for v in verts_pq_orig]
    
    # mapping from plane coords to image plane
    M_uv_to_xy = np.matrix([
        [focal_pixels, 0, 0],
        [0, focal_pixels, 0],
        [0, 0, -1]
    ]) * np.matrix([
        [uvnb[0], uvnb[4], uvnb[12]],
        [uvnb[1], uvnb[5], uvnb[13]],
        [uvnb[2], uvnb[6], uvnb[14]]
    ])
    
    M_xy_to_uv = np.linalg.inv(M_uv_to_xy)

    M_pq_to_xy = np.matrix([
        [1, 0, -0.5 * w],
        [0, -1, 0.5 * h],
        [0, 0, 1],
    ])

    # estimate rough resolution from original bbox
    min_p, min_q, max_p, max_q = bbox_vertices(verts_pq)
    max_dim = max(max_p - min_p, max_q - min_q)

    # transform
    verts_xy = transform(M_pq_to_xy, verts_pq)
    verts_uv = transform(M_xy_to_uv, verts_xy)

    # compute bbox in uv plane
    min_u, min_v, max_u, max_v = bbox_vertices(verts_uv)
    max_uv_range = float(max(max_u - min_u, max_v - min_v))

    # scale so that st fits inside [0, 1] x [0, 1]
    # (but with the correct aspect ratio)
    M_uv_to_st = np.matrix([
        [1, 0, -min_u],
        [0, -1, max_v],
        [0, 0, max_uv_range]
    ])

    verts_st = transform(M_uv_to_st, verts_uv)

    M_st_to_ij = np.matrix([
        [max_dim, 0, 0],
        [0, max_dim, 0],
        [0, 0, 1]
    ])

    verts_ij = transform(M_st_to_ij, verts_st)

    # find final bbox
    min_i, min_j, max_i, max_j = bbox_vertices(verts_ij)
    size = (int(math.ceil(max_i)), int(math.ceil(max_j)))

    # homography for final pixels to original pixels (ij --> pq)
    M_pq_to_ij = M_st_to_ij * M_uv_to_st * M_xy_to_uv * M_pq_to_xy
    M_ij_to_pq = np.linalg.inv(M_pq_to_ij)
    M_ij_to_pq /= M_ij_to_pq[2, 2]  # NORMALIZE!
    data = M_ij_to_pq.ravel().tolist()[0]
    rectified = photo_image.transform(
        size=size, method=Image.PERSPECTIVE,
        data=data, resample=Image.BICUBIC)
    return rectified, verts_ij

