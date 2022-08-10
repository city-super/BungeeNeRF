import numpy as np
import math
import os
import json
import imageio
from sklearn.metrics import pairwise_distances

def get_intrinsic(imgdir):
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]

    H, W, C = imageio.imread(imgfiles[0]).shape
    vfov = 40

    focal = H / 2  / np.tan(np.deg2rad(vfov/2))

    return H, W, focal

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def pad_rot(rot):
    padh = lambda x: np.hstack([x, np.zeros((x.shape[0], 1))])
    padv = lambda x: np.vstack([x, np.zeros((1, x.shape[1]))])

    rot_mat = padv(padh(rot))
    rot_mat[-1,-1] = 1
    return rot_mat


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default='data/multiscale_google_56Leonard', help='path to your meta')
    parser.add_argument("--latlng", type=lambda s: [float(item) for item in s.split(',')], default=[40.71761215662889,-74.00627852686617], help='latlng of center building')
    parser.add_argument("--scale_split", type=lambda s: [int(item) for item in s.split(',')], default=[275,150,70,0], help='split index of each scale, mannually set for now')
    
    return parser



if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()
 
    with open(os.path.join(args.datadir, 'GES_local.json'), 'r') as f:
        data = json.load(f)

    GES_pos = np.array([[data['cameraFrames'][i]['position']['x'], 
                            data['cameraFrames'][i]['position']['y'],
                            data['cameraFrames'][i]['position']['z']] 
                        for i in range(len(data['cameraFrames']))])

    H, W, focal = get_intrinsic(os.path.join(args.datadir, 'images'))

    # rescale the whole range if you want
    scale = 2**3 * np.pi / max(GES_pos.max(), -GES_pos.min())
    SS = np.eye(4)
    SS[0,0] = scale
    SS[1,1] = scale
    SS[2,2] = scale
    
    rclat, rclng = np.radians(args.latlng[0]), np.radians(args.latlng[1]) 
    rot_ECEF2ENUV = np.array([[-math.sin(rclng),                math.cos(rclng),                              0],
                              [-math.sin(rclat)*math.cos(rclng), -math.sin(rclat)*math.sin(rclng), math.cos(rclat)],
                              [math.cos(rclat)*math.cos(rclng),  math.cos(rclat)*math.sin(rclng),  math.sin(rclat)]])

    nxyz = []
    poses = []
    for i in range(len(data['cameraFrames'])):
        position = data['cameraFrames'][i]['position']
        pos_x = position['x']
        pos_y = position['y']
        pos_z = position['z']
        xyz = np.array([pos_x, pos_y, pos_z])
        [pos_e,pos_n,pos_u] = np.dot(rot_ECEF2ENUV, xyz)

        rotation = data['cameraFrames'][i]['rotation']

        x = np.radians(-rotation['x'])
        y = np.radians(180-rotation['y'])
        z = np.radians(180+rotation['z'])

        rot_mat = np.linalg.inv(eulerAnglesToRotationMatrix([x, y, z]))
        rot_mat = np.dot(rot_ECEF2ENUV, rot_mat)
        GES_rotmat = pad_rot(rot_mat)

        xyz  = np.array([pos_e,pos_n,pos_u,1])[None,:]
        nx,ny,nz = np.dot(SS, xyz.T)[:3,0]
        nxyz.append([nx,ny,nz])
        GES_rotmat[:3,3] = np.array([nx,ny,nz])

        c2w = (np.hstack([GES_rotmat[:3,:4], np.array([[H, W, focal]]).T]))
        poses.append(c2w)

    nxyz = np.array(nxyz)
    dists = np.sqrt(np.sum(nxyz**2, -1))

    # heuristic bds (NOT USED IN TRAINING)
    dists = pairwise_distances(np.array(poses)[:,:,3])
    dists_flatten = dists.flatten()
    self_idx = [i*len(dists)+i for i in range(len(dists))]
    new_dists = np.array([dists_flatten[i] for i in range(len(dists)*len(dists)) if i not in self_idx])
    near = max(new_dists.min(), 1e-10)
    far = 1.1 * new_dists.max()


    poses = np.array(poses).reshape(-1, 3*5)
    bds = np.array([[near, far] for _ in range(len(poses))])
    
    poses_bounds = np.hstack([poses, bds])

    cam_pose = {'poses' : poses_bounds.tolist(),
                'scene_scale' : scale,
                'scene_origin' : [0., 0., -6371011.], # earth center is fixed in ENU coord
                'scale_split' : args.scale_split,} 

    with open(os.path.join(args.datadir, 't_poses_enu.json'), 'w') as f: 
        json.dump(cam_pose, f)
        