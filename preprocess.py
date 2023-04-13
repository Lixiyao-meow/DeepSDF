from mesh_to_sdf import sample_sdf_near_surface

import os
import glob
import trimesh
import pyrender
import numpy as np


def generate_xyz_sdf(filename):
    mesh = trimesh.load(os.path.abspath(filename),force='mesh')
    xyz, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
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
    

class_path = "/04256520/"

# make directory in target file
target_path = "./data" + class_path

isExist = os.path.exists(target_path)

if not isExist:
   os.makedirs(target_path)

# find all mesh file from dataset
mesh_filenames = list(glob.iglob("dataset" + class_path + "/**/*.obj"))

N = len(mesh_filenames)
it = 0

for mesh_filepath in mesh_filenames:
    
    list_mesh_filepath = mesh_filepath.split("/")
    target_filepath = os.path.join(target_path, list_mesh_filepath[2])
        
    target_filepath = os.path.join(target_filepath, list_mesh_filepath[3].split(".")[0])
    print(target_filepath)
    # generate point clounds
    #process(mesh_filepath, target_filepath)

    it += 1
    print("process finished:", list_mesh_filepath[2], it, "/", N)