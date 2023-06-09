# DeepSDF

We implemented an implicit representation, the Signed Distance Function (SDF) representation, based on the paper by Park et al. (2019) (https://arxiv.org/abs/1901.05103).

Our objective was to reproduce the auto-decoder for a batch of similar shapes, and then train and test the network on the ShapeNet dataset. We trained our model on 50 samples from the sofa category for 200 epochs.

## Run

### Preprocessing
To preprocess the data, run ```preprocess.py```. We used the Python library ```mesh_to_sdf``` to sample 15,000 points non-uniformly near the surface. By default, we set the path to ```./processed_data/validation/```. However, note that the user will need to set the path to ```./processed_data/train/``` to process training data.

### Train
To train the model, run ```train.py```. You will need to modify the training parameters directly in this file.
We saved our trained model in ```checkpoints/trained_model.pt```.

### Reconstruction
To reconstruct the mesh, run ```reconstruct.py```. This file will load each point cloud from ```./processed_data/validation/``` and infer the SDF value to render the zero-level-set boundary using the marching cubes algorithm.

## Result

### Preprocessing

The points are coloured with their SDF values with a linear scale from red (+) to blue (-). Most of the sampled points are close to the surface, while only a few points are outside in the unit sphere.

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/point_clouds.jpeg" width="720">

### Training loss

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/training_loss.png" width="540">

### Shape reconstruction

The upper graphs show the reconstructed shape, while the bottom graphs show the ground truth. Although our model is able to learn the general shape of the class and some variation can be observed depending on the input model’s shape, it fails to capture more intricate details. 

<img src="https://github.com/Lixiyao-meow/DeepSDF/blob/main/img/reconstruction.png" width="720">

Based on our experiments, we believe that the SDF is a promising representation, but the idea of removing the encoder does not seem reliable. More analysis and discussion can be found in our report in this repository.

## References

- DeepSDF: https://github.com/facebookresearch/DeepSDF
- Python Library ```mesh-to-sdf```: https://pypi.org/project/mesh-to-sdf/
