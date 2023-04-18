# DeepSDF

We implemented an implicit representation, the Signed Distance Function (SDF) representation, based on the paper by Park et al. (2019) (https://arxiv.org/abs/1901.05103).

Our objective was to reproduce the auto-encoder for a batch of similar shapes, and then train and test the network on the ShapeNet dataset. We trained our model on 50 samples from the sofa category for 200 epochs.

## Run

### Preprocessing
To preprocess the data, run ```preprocess.py```. We used the Python library ```mesh_to_sdf``` to sample 15,000 points non-uniformly near the surface. Note that the user will need to set the path to the dataset in this file.

### Train
To train the model, run ```train.py```. You will need to modify the training parameters directly in this file.

### Reconstruction
To reconstruct the mesh, run ```reconstruct.py```. This file will load each point cloud from ```./processed_data/validation/``` and infer the SDF value to render the zero-level-set boundary using the marching cubes algorithm.

## Result

### Preprocessing

The points are coloured with their SDF values with a linear scale from red (+) to blue (-). Most of the sampled points are close to the surface, while only a few points are outside in the unit sphere.

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/point_clouds.jpeg" width="720">

### Training loss

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/training_loss.png" width="360">

### Shape reconstruction

Although our model is able to learn the general shape of the class and some variation can be observed depending on the input model’s shape, it fails to capture more intricate details.

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/reconstruction.png" width="720">

