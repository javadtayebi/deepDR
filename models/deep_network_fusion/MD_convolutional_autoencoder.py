from tensorflow.keras import layers, Sequential, Model, optimizers, losses, utils, metrics


def encode(inputs, input_dims, encoder_type, dense_layers_architecture):
    # First encoding layer(s)
    first_encoder = Sequential()

    first_encoder.add(
        layers.Conv1D(
            filters=1,
            kernel_size=100,
            strides=1,
            padding='valid',
            activation='sigmoid'
        )
    )
    first_encoder.add(
        layers.AveragePooling1D(pool_size=2)
    )

    # If encoder_type == 1, then: before concatenation, outputs should be flatten:
    if encoder_type == 1:
        first_encoder.add(
            layers.Flatten()
        )
    # Apply First encoding layer(s), on inputs
    out = []
    for i in range(0, len(input_dims)):
        out.append(
            first_encoder(inputs[i])
        )

    # Concatenating outputs of first encoding layer
    concatenated = layers.concatenate(out, axis=1)

    # Encoding layer: after Concatenation, before middle layer
    encoder = Sequential()

    # If encoder_type == 1, then: after concatenation, layers should be dense with desired arch. :
    if encoder_type == 1:
        for j in range(1, len(dense_layers_architecture) // 2):
            encoder.add(
                layers.Dense(dense_layers_architecture[j],
                             activation='sigmoid')
            )
    # Else, after concatenation, layers should be convolutional too:
    else:
        encoder.add(
            layers.Conv1D(
                filters=1,
                kernel_size=100,
                strides=1,
                padding='valid',
                activation='sigmoid'
            )
        )
        encoder.add(
            layers.AveragePooling1D(pool_size=2)
        )

    # Middle layer
    flatten = layers.Flatten()

    middle_layer = layers.Dense(
        dense_layers_architecture[len(dense_layers_architecture) // 2],
        activation='sigmoid',
        name="middle_layer")(encoder(concatenated) if encoder_type == 1 else flatten(encoder(concatenated)))

    return middle_layer


def decode(inputs, input_dims, dense_layers_architecture):
    # Decoding layer: after middle layer, before last decoding layer
    decoder = Sequential()
    for k in range((len(dense_layers_architecture) // 2) + 1, len(dense_layers_architecture)):
        decoder.add(
            layers.Dense(dense_layers_architecture[k],
                         activation='sigmoid')
        )

    # Last decoding layer
    out = []
    for _ in range(0, len(input_dims)):
        out.append(
            layers.Dense(dense_layers_architecture[-1] // len(input_dims),
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


def multimodal_deep_convolutional_autoencoder(input_dims, encoder_type, dense_layers_architecture):
    # Input layers
    inputs = []
    for dim in input_dims:
        inputs.append(layers.Input(shape=(dim, 1)))

    # Middle layer
    middle_layer = encode(
        inputs=inputs,
        input_dims=input_dims,
        encoder_type=encoder_type,
        dense_layers_architecture=dense_layers_architecture
    )

    # Output layers
    outputs = decode(
        inputs=middle_layer,
        input_dims=input_dims,
        dense_layers_architecture=dense_layers_architecture
    )

    # Autoencoder model
    sgd_optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False)
    adam_optimizer = 'adam'

    model = Model(inputs=inputs, outputs=outputs)
    # encoder_model = Model(inputs=inputs, outputs=middle_layer)

    model.compile(
        optimizer=sgd_optimizer,
        loss=losses.BinaryCrossentropy()
    )

    print(model.summary())

    utils.plot_model(
        model=model,
        to_file='./models_plots/MD_convolutional_autoencoder.png',
        show_shapes=True
    )

    return model
