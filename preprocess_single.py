from mesh_to_sdf import sample_sdf_near_surface

import os
import glob
import trimesh
import pyrender
import numpy as np


def generate_xyz_sdf(filename):
    mesh = trimesh.load(os.path.abspath(filename),force='mesh')
    xyz, sdf = sample_sdf_near_surface(mesh, number_of_points=15000)
    return xyz, sdf

def writeSDFToNPZ(xyz, sdfs, filename):
    num_vert = len(xyz)
    pos = []
    neg = []

    for i in range(num_vert):
        v = xyz[i]
        s = sdfs[i]

        if s > 0:
            for j in range(3):
                pos.append(v[j])
            pos.append(s)
        else:
            for j in range(3):
                neg.append(v[j])
            neg.append(s)
    
    np.savez(filename, pos=np.array(pos).reshape(-1, 4), neg=np.array(neg).reshape(-1, 4))
    
def process(mesh_filepath, target_filepath):
    xyz, sdfs = generate_xyz_sdf(mesh_filepath)
    writeSDFToNPZ(xyz, sdfs, target_filepath)
    

file_path = "./dataset/04256520/train/1.obj"

# make directory in target file
target_path = "./processed_data/04256520/single/"

repeat = 10

for i in range(repeat):
    
    target_filepath = os.path.join(target_path, str(i+1))
    
    # generate point clounds
    process(file_path, target_filepath)

    print("process finished:", i+1, "/", repeat)