import os
import pickle
import sys
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
from tensorflow.keras import callbacks, Model, utils
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from models.deep_network_fusion.MD_autoencoder import multimodal_deep_autoencoder
from models.deep_network_fusion.MD_variational_autoencoder import multimodal_deep_variational_autoencoder, Sampling
from models.deep_network_fusion.MD_convolutional_autoencoder import multimodal_deep_convolutional_autoencoder
from models.deep_network_fusion.MD_convolutional_variational_autoencoder import multimodal_deep_convolutional_variational_autoencoder
from utils import read_params, prepare_inputs

"""
Main code starts here
"""
params = read_params(sys.argv[1])
# print(params)

ae_type = params['ae_type']  # Type of autoencoder
select_arch = params['select_arch']  # A number in range 1-10
select_nets = params['select_nets']  # A list of selected networks
epochs = params['epochs']
batch_size = params['batch_size']
noise_factor = params['noise_factor']  # nf > 0 for for corrupting the input

models_path = './test_models/'
results_path = './models_plots/results/'

"""
Possible autoencoder architectures
"""
n = len(select_nets)
regular_archs = {
    1: [n * 1000, n * 100, n * 1000],
    2: [n * 1000, n * 500, n * 100, n * 500, n * 1000],
    3: [n * 1000, n * 500, n * 200, n * 100, n * 200, n * 500, n * 1000],
    4: [n * 1000, n * 800, n * 500, n * 200, n * 100, n * 200, n * 500, n * 800, n * 1000],
}

conv_archs_1 = {
    1: [n * 100, n * 1000],
    2: [n * 500, n * 100, n * 500, n * 1000],
    3: [n * 500, n * 200, n * 100, n * 200, n * 500, n * 1000],
    4: [n * 800, n * 500, n * 200, n * 100, n * 200, n * 500, n * 800, n * 1000],
}

conv_archs_2 = {
    1: [n * 100, n * 1000],
    2: [n * 100, n * 500, n * 1000],
    3: [n * 100, n * 200, n * 500, n * 1000],
    4: [n * 100, n * 200, n * 500, n * 800, n * 1000],
}

"""
load PPMI matrices
"""
Nets = []
input_dims = []
for i in select_nets:
    print("### [%d] Loading network..." % (i))
    N = sio.loadmat('./PPMI/' + 'drug' + '_net_' + str(i) + '.mat', squeeze_me=True)
    Net = N['Net'].todense()
    Net = np.asarray(Net)
    print("Net %d, NNofile_keywords=%d \n" % (i, np.count_nonzero(Net)))
    Nets.append(minmax_scale(Net))
    input_dims.append(Net.shape[1])

"""
Training Multimodal Deep Autoencoder
"""
print(f"### Preparing model inputs...")

X_train, X_train_noisy, X_test, X_test_noisy = prepare_inputs(X=Nets, noise_factor=noise_factor, std=1.0)

print(f"### [Multimodal Deep {ae_type} Autoencoder] is Running:")

if ae_type == 'convolutional':
    model = multimodal_deep_convolutional_autoencoder(
        input_dims=input_dims,
        encoder_type=2,
        dense_layers_architecture=conv_archs_2[select_arch]
    )
elif ae_type == 'variational':
    model = multimodal_deep_variational_autoencoder(
        input_dims=input_dims,
        autoencoder_architecture=regular_archs[select_arch]
    )
elif ae_type == 'convolutional_variational':
    model = multimodal_deep_convolutional_variational_autoencoder(
        input_dims=input_dims
    )
else:
    model = multimodal_deep_autoencoder(
        input_dims=input_dims,
        autoencoder_architecture=regular_archs[select_arch]
    )

# Fitting the model
if ae_type == 'variational' or ae_type == 'convolutional_variational':
    history = model.fit(
        x=X_train,
        y=X_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=500)],
        validation_data=(X_test, X_test),
        shuffle=True
    )
else:
    history = model.fit(
        x=X_train_noisy,
        y=X_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=500)],
        validation_data=(X_test_noisy, X_test),
        shuffle=True
    )

encoder_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)
# Save encoder_model
if ae_type == 'convolutional':
    model_name = f"CAE_model"
    encoder_model_name = f"CAE_encoder_model.h5"
elif ae_type == 'variational':
    model_name = f"VAE_model"
    encoder_model_name = f"VAE_encoder_model.h5"
elif ae_type == 'variational_convolutional':
    model_name = f"CVAE_model"
    encoder_model_name = f"CVAE_encoder_model.h5"
else:
    model_name = f"AE_model"
    encoder_model_name = f"AE_encoder_model.h5"

encoder_model.save(models_path + encoder_model_name)

with open(models_path + model_name + '_history.pckl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Export figure: loss vs epochs (history)
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Multimodal Deep ' + ae_type + 'Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.savefig(results_path + model_name + '_loss.png', bbox_inches='tight')

# auc, val_auc = compute_metric_mean(history=history, metric='auc')
# # Export figure: auc vs epochs (history)
# plt.figure()
# plt.plot(auc, label='Training auc')
# plt.plot(val_auc, label='Validation auc')
# plt.title('Multimodal Deep ' + ae_type + 'Autoencoder model auc')
# plt.ylabel('auc')
# plt.xlabel('epoch')
# plt.legend(loc='upper left')
# plt.grid()
# plt.savefig(results_path + model_name + '_auc.png', bbox_inches='tight')

"""
Saving features
"""
print(f"### Running for: {encoder_model_name}")

if not os.path.isfile(models_path + encoder_model_name):
    print(f"### Model {encoder_model_name} does not exist. Check the 'models_path' directory.\n")
else:
    if ae_type == 'variational':
        with utils.CustomObjectScope({'Sampling': Sampling}):
            encoder_model = load_model(models_path + encoder_model_name, compile=False)
    else:
        encoder_model = load_model(models_path + encoder_model_name, compile=False)

    features = encoder_model.predict(Nets)
    features = minmax_scale(features)

    features_file_name = 'drug_features.txt'
    np.savetxt(features_file_name, features, delimiter='\t', fmt='%s', newline='\n')
    print(
        f"### It's Done!\nExtracted features from middle layer of autoencoder are saved as {features_file_name} "
        f"in project's root directory.")
