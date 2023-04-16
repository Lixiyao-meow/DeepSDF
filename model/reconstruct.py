import torch

import numpy as np
import plyfile
import skimage.measure

# Learn the latent code for the test model
def reconstruct_latent(decoder, 
                       sdf_data, 
                       iterations = 800,
                       init_std = 0.01, 
                       lr = 5e-4):

    # parameters
    latent_size = 256
    num_samples = 15000

    # initilise latent
    latent = torch.ones(1, latent_size).normal_(mean=0, std=init_std)
    latent.requires_grad = True

    # set optimizer and loss
    optimizer = torch.optim.Adam([latent], lr=lr)
    loss_l1 = torch.nn.L1Loss()
    minT, maxT = -0.1, 0.1 # clamp

    for it in range(iterations):
        
        decoder.eval()

        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, minT, maxT)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, xyz], 1)
        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

        loss = loss_l1(pred_sdf, sdf_gt)
        loss.backward()
        optimizer.step()
        
        loss_num = loss.cpu().data.numpy()
        print('[%d/%d] Loss: %.5f' % (it+1, iterations, loss_num))

    return latent

# Predict SDF with the latent code and trained decoder
def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)

    inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)
    
    return sdf

'''
Rebuild mesh from SDF predictions
This function is adapted from: https://github.com/facebookresearch/DeepSDF
'''
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

def create_mesh(filename, 
                decoder, 
                latent_vec,
                N=128, 
                max_batch=16 ** 3):
 

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]#.cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        filename + ".ply"
    )


def reconstruct(test_sample,
                decoder,
                filename,
                lat_iteration,
                lat_init_std = 0.01, 
                lat_lr = 5e-4,
                N=128, 
                max_batch=16 ** 3):

    # pass the test model to decoder
    _, test_sdf_data = test_sample

    print("---- Fitting latent vector ----")
    latent_vector = reconstruct_latent(decoder, 
                                       test_sdf_data, 
                                       iterations = lat_iteration,
                                       init_std = lat_init_std, 
                                       lr = lat_lr)

    print("---- Reconstructing mesh ----")
    print(" This could take a while ")
    
    create_mesh(filename, 
                decoder,
                latent_vector,
                N=N, 
                max_batch=max_batch)
    
    print("Mesh saved to " + filename + ".ply")