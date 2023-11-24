from tensorflow.keras import layers, Sequential, Model, optimizers, losses, utils, metrics


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

    # Middle layer
    middle_layer = layers.Dense(autoencoder_architecture[len(autoencoder_architecture) // 2],
                                activation='sigmoid',
                                name="middle_layer")(encoder(concatenated))

    return middle_layer


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


def multimodal_deep_autoencoder(input_dims, autoencoder_architecture):
    # Input layers
    inputs = []
    for dim in input_dims:
        inputs.append(layers.Input(shape=(dim,)))

    # Middle layer
    middle_layer = encode(inputs=inputs, input_dims=input_dims, autoencoder_architecture=autoencoder_architecture)

    # Output layers
    outputs = decode(inputs=middle_layer, input_dims=input_dims, autoencoder_architecture=autoencoder_architecture)

    # Autoencoder model
    sgd_optimizer = optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False)
    adam_optimizer = 'adam'

    model = Model(inputs=inputs, outputs=outputs)
    # encoder_model = Model(inputs=inputs, outputs=middle_layer)

    model.compile(
        optimizer=sgd_optimizer,
        loss=losses.BinaryCrossentropy(),
    )

    print(model.summary())

    utils.plot_model(
        model=model,
        to_file='./models_plots/MD_autoencoder.png',
        show_shapes=True
    )

    return model
