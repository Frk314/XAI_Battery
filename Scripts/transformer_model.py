import tensorflow as tf
from tensorflow.keras import layers, models, Input
import keras
import os
from keras import layers
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom
import math, pickle,glob,os
from reportlab.pdfgen import canvas
from matplotlib.backends.backend_pdf import PdfPages

import sys
from batteryml.pipeline import Pipeline, load_config, build_dataset
from batteryml.visualization.plot_helper import plot_capacity_degradation, plot_cycle_attribute, plot_result

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')
TF_ENABLE_ONEDNN_OPTS=0

###########  Data Loading  ################
pipeline = Pipeline(config_path='configs/baselines/nn_models/transformer/matr_1.yaml',
                    workspace='workspaces')
configs = load_config(config_path='configs/baselines/nn_models/transformer/matr_1.yaml', workspace='workspaces')
dataset, raw_data = build_dataset(configs, device='cpu')
###########################


#We convert the data into tensor and extract 10% of the record that we are going to use
def to_tf(tensor):
    tensor_array = tensor.cpu().numpy()
    with tf.device('/CPU:0'):
        tensor_tf = tf.constant(tensor_array)
    return tensor_tf
 
#We reshape our data
def reshape_data(a):
    b = []
    for i in range(len(a)):
        # Use np.transpose to swap the second and third dimensions
        b.append(np.transpose(a[i, :, :], (1, 0)))

    # Convert list of arrays to a NumPy array
    b = np.array(b)
    return(b)

train_data = dataset.train_data.feature
train_label = dataset.train_data.label

train_data_tf = to_tf(train_data)
train_label_tf = to_tf(train_label)
train_data_tf = reshape_data(train_data_tf)
# Extract every 10th element from the second axis (axis 1)
train_data_tf = train_data_tf[:, ::10, :]
train_data_tf = np.reshape(train_data_tf, (-1, 100, 100, 1))
train_label_tf = np.reshape(train_label_tf, (-1,1))

#we separate train and validation data
val = int(0.25*len(train_data_tf))

train_data_tf, val_data_tf = tf.split(train_data_tf, num_or_size_splits=[len(train_data_tf)-val,val], axis=0)
train_label_tf, val_label_tf = tf.split(train_label_tf, num_or_size_splits=[len(train_label_tf)-val,val], axis=0)


######################### CLASSES AND FUNCTIONS  ###########################
# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class AttentionBlock(keras.layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = keras.layers.GroupNormalization(groups=groups)
        self.query = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = keras.layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("blc, bLc->blL", q, k) * scale
        attn_score = tf.nn.softmax(attn_score, -1)

        proj = tf.einsum("blL,bLc->blc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj, attn_score

def TransformerEncoder(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        pemb = keras.layers.Reshape((-1, 100))(inputs)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)

        pemb, attn_score = AttentionBlock(units, groups=8)(pemb)
        pemb = keras.layers.GlobalAveragePooling1D()(pemb)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)
        return pemb, attn_score
    
    return apply

def get_prediction(data, model):
    prediction = model.predict(data)
    prediction = tf.reshape(prediction, (1,len(data)))
    return dataset.evaluate(torch.from_numpy(prediction.numpy()), 'RMSE', data_type='test')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = np.reshape(img_array, (1, 100, 100))
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output[0], model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel , last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads)

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.abs(tf.squeeze(heatmap))

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()


def make_model(input_shape=(100, 100), num_classes=1):
    # Input layer
    inputs = Input(shape=input_shape)

    # Entry block
    x, s = TransformerEncoder(32)(inputs)

    # Output layer
    outputs = layers.Dense(num_classes)(x)

    # Create the model using functional API
    model = models.Model(inputs, outputs)

    return model


model = make_model(input_shape=(100, 100))

epochs = 1000

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    ]

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="mean_squared_error", 
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)

history = model.fit(
    train_data_tf,
    train_label_tf,
    epochs=epochs,
    validation_data=(val_data_tf, val_label_tf)
    )
    
# #with PdfPages('{}/history.pdf'.format(m)) as pdf:
# plt.figure(figsize= (16,5))
# plt.plot(history.history["root_mean_squared_error"][5:], color="r")
# plt.plot(history.history['val_root_mean_squared_error'][5:], color="b")
# plt.title("Model RMSE")
# plt.xlabel("epochs") 
# #pdf.savefig()

# plt.show()

test_data = dataset.test_data.feature
test_label = dataset.test_data.label

test_data_tf = to_tf(test_data)
test_label_tf = to_tf(test_label)
test_data_tf = reshape_data(test_data_tf)
test_data_tf = test_data_tf[:, ::10, :]

test_label_tf = np.reshape(test_label_tf, (-1, 1))
test_data_tf = np.reshape(test_data_tf, (-1, 100, 100, 1))

last_conv_layer_name = model.layers[3].name
m="model_1"

with PdfPages('{}/Result.pdf'.format(m)) as pdf:
    preds, pred_table = get_prediction(test_data_tf, model)[0] # get_prediction format: ((prediction, list of predictions), )
    
    #plt.figure(figsize= (12,5))
    #plt.plot(history.history["root_mean_squared_error"][5:], color="r")
    #plt.plot(history.history['val_root_mean_squared_error'][5:], color="b")
    #plt.title("Model RMSE")
    #plt.xlabel("epochs")
    #pdf.savefig()
    #plt.close()

    plt.figure(figsize= (12,1))
    plt.text(0.5, 0.5, 'RMSE of the model: {}'.format(preds , ha='center', va='center', fontsize=10))
    plt.axis('off')
    # pdf.savefig()
    plt.close()    
    
    for i, data in enumerate(test_data_tf):
        fig = plt.figure(figsize=(10, 5))  # Create a new figure for each plot
        plt.text(0.1, 0.05, abs(pred_table.numpy()[0][i]) , ha='center', va='center', fontsize=10)
        plt.axis('off')
        
        
        # First subplot: display the data with imshow
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(data)  # Plot the data
        ax1.set_title(f"Qmatrix of the Cell {i+1}")  # Title for the first plot
        ax1.axis('off')  # Turn off axis for the first subplot
        cbar1 = plt.colorbar(im1, shrink=0.75, aspect=20, pad=0.05)

        heatmap = make_gradcam_heatmap(test_data_tf[i], model, last_conv_layer_name)
        ax2 = fig.add_subplot(122)
        # Resize heatmap to (100, 100) using interpolation
        resized_heatmap = zoom(heatmap, (1, 100 / 32), order=1)  # Linear interpolation
        im2 = ax2.matshow(resized_heatmap, vmin=0, vmax=1, cmap='jet')  # Display the heatmap
        cbar2 = plt.colorbar(im2, shrink=0.75, aspect=20, pad=0.05)
        ax2.set_title("Grad-CAM Heatmap")  # Title for the heatmap
        ax2.axis('off')  # Turn off axis for the second subplot
    
        plt.tight_layout()  # Adjust layout to prevent overlap
        pdf.savefig()
        plt.show()  # Display the figure
        #plt.pause(3)  # Pause for 3 seconds before showing the next plot

        with open('{}/heatmap_{}.pkl'.format(m, i), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(heatmap, f)
