import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, optimizers, losses, utils, metrics


class Sampling(layers.Layer):
    """
    Use re-parameterization trick for sampling z
    """

    def call(self, inputs, *args, **kwargs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=(tf.shape(mean)[0], tf.shape(mean)[1]))

        # use exp(0.5*log_var) instead of stddev:
        z = mean + tf.exp(0.5 * log_var) * epsilon

        return z


def encode(inputs, input_dims, autoencoder_architecture):
    # First encoding layer
    out = []
    for i in range(0, len(input_dims)):
        out.append(
            layers.Dense(autoencoder_architecture[0] // len(input_dims),
                         activation='sigmoid')(inputs[i])
        )

    # Concatenating outputs of first encoding layer
    concatenated = layers.concatenate(out)

    # Encoding layer: after Concatenation, before middle layer
    encoder = Sequential()
    for j in range(1, len(autoencoder_architecture) // 2):
        encoder.add(
            layers.Dense(autoencoder_architecture[j],
                         activation='sigmoid')
        )

    latent_space_dim = autoencoder_architecture[len(autoencoder_architecture) // 2]
    mean = layers.Dense(latent_space_dim, name="mean")(encoder(concatenated))
    log_var = layers.Dense(latent_space_dim, name="log_var")(encoder(concatenated))

    # Middle layer(Latent space)(z)
    z = Sampling(name='middle_layer')([mean, log_var])

    return mean, log_var, z


def decode(inputs, input_dims, autoencoder_architecture):
    # Decoding layer: after middle layer, before last decoding layer
    decoder = Sequential()
    for k in range((len(autoencoder_architecture) // 2) + 1, len(autoencoder_architecture)):
        decoder.add(
            layers.Dense(autoencoder_architecture[k],
                         activation='sigmoid')
        )

    # # Reconstruction of the concatenated layer
    # concatenated_reconstruct = layers.Dense(autoencoder_architecture[0],
    #                                         activation='sigmoid')(decoder(inputs))

    # Last decoding layer
    out = []
    for _ in range(0, len(input_dims)):
        out.append(
            layers.Dense(autoencoder_architecture[-1] // len(input_dims),
                         activation='sigmoid')(decoder(inputs))
        )

    # Data reconstruction
    outputs = []
    for m in range(0, len(input_dims)):
        outputs.append(
            layers.Dense(input_dims[m],
                         activation='sigmoid')(out[m])
        )

    return outputs


def multimodal_deep_variational_autoencoder(input_dims, autoencoder_architecture):
    # Input layers
    inputs = []
    for dim in input_dims:
        inputs.append(layers.Input(shape=(dim,)))

    # Middle layer(Latent space)
    mean, log_var, z = encode(inputs=inputs, input_dims=input_dims, autoencoder_architecture=autoencoder_architecture)

    # Output layers
    outputs = decode(inputs=z, input_dims=input_dims, autoencoder_architecture=autoencoder_architecture)

    # Autoencoder model
    sgd_optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False)
    adam_optimizer = 'adam'

    model = Model(inputs=inputs, outputs=outputs)
    # encoder_model = Model(inputs=inputs, outputs=middle_layer)

    print(model.summary())

    utils.plot_model(
        model=model,
        to_file='./models_plots/MD_variational_autoencoder.png',
        show_shapes=True
    )

    # Add KL divergence loss
    kl_divergence_loss = -0.5 * tf.reduce_mean(
        log_var - tf.square(mean) - tf.exp(log_var) + 1
    )
    model.add_loss(kl_divergence_loss)

    model.compile(
        optimizer=adam_optimizer,
        loss=losses.BinaryCrossentropy(),
    )

    return model
