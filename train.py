import model.train as train

# train step
train.train_decoder(epochs = 200,
                    batch_size=5,
                    lat_vecs_std = 0.01,
                    decoder_lr = 0.0005,
                    lat_vecs_lr = 0.001)