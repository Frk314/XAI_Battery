#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt

from tensorflow.keras import layers, models, Input
from scipy.ndimage import zoom
import math, pickle,glob,os, random, torch
from reportlab.pdfgen import canvas
from matplotlib.backends.backend_pdf import PdfPages


import sys
from batteryml.pipeline import Pipeline, load_config, build_dataset
from batteryml.visualization.plot_helper import plot_capacity_degradation, plot_cycle_attribute, plot_result


# Create a pipeline with a configuration file, specifying the device and workspace. 
# Developers need to modify the data, feature, model and other related settings in the config file in advance. 
pipeline = Pipeline(config_path='configs/baselines/nn_models/transformer/matr_1.yaml',
                    workspace='workspaces')

configs = load_config(config_path='configs/baselines/nn_models/transformer/matr_1.yaml', workspace='workspaces')
dataset, raw_data = build_dataset(configs, device='cpu')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')




#We convert the data into tensor and extract 10% of the record that we are going to use
def to_tf(tensor):
    tensor_array = tensor.numpy()
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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # Ensure the model has been built (called with input data)
    
    img_array = np.reshape(img_array, (1, 100, 100))
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output[0], model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
         
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
        
        grads = tape.gradient(class_channel , last_conv_layer_output)
        
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.abs(tf.squeeze(heatmap))

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def compile_model(model):
    
    model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
    return model

def get_prediction(data, model):
    with tf.device('/CPU:0'):
            prediction = model.predict(data)
            prediction = tf.reshape(prediction, (1,len(data)))
            return dataset.evaluate(torch.from_numpy(prediction.numpy()), 'RMSE', data_type='test')
    
def reset_model():
    mod = make_model(input_shape=(100, 100, 1))
    mod.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    return mod


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

class GradCAMActivationCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample, last_conv_layer_name, threshold):
        """
        Args:
            sample (ndarray): A batch of training images (e.g., 10 images).
            last_conv_layer_name (str): The name of the convolutional layer to inspect.
            threshold (float): The activation threshold.
        """
        super().__init__()
        self.sample = sample  # Expecting a batch of images
        self.last_conv_layer_name = last_conv_layer_name
        self.threshold = threshold
        self.stopped_due_to_threshold = False

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and epoch % 499 == 0:  # Run only every 500 epochs
            print(f"Epoch {epoch} - Checking Grad-CAM activations")
            perf_values = []

            for idx, img in enumerate(self.sample):
                heatmap = make_gradcam_heatmap(img, model, self.last_conv_layer_name)

                # Select region of interest (ROI)
                roi_1 = heatmap[20:80, 18:]  # Extract the region with the charge voltage depression
                roi_2 = heatmap[20:80, :18]  # Extract the region adjacent to it for comparisom

                # Compute the average activation in this region
                avg_act_1 = np.mean(roi_1)
                avg_act_2 = np.mean(roi_2)

                perf = avg_act_1 / (avg_act_1 + avg_act_2)

                perf_values.append(perf)

                print(f"Epoch {epoch}: Sample {idx} - ROI average activation = {perf:.4f}")

            perf_avg = 0
            if perf_values:  # Ensure non-empty list before computing overall average
                self.threshold = perf_avg if perf_avg > 0.5 else self.threshold # ensure progression in the model training
                perf_avg = np.mean(perf_values)
                print(f"Epoch {epoch}: Performance of the model = {perf_avg:.4f}")
                 
             
                if perf_avg < self.threshold or np.isnan(perf_avg):
                    print(f"Threshold not reached (avg {perf_avg:.4f} < {self.threshold}). Stopping training.")
                    self.stopped_due_to_threshold = True
                    self.model.stop_training = True
            else:
                print(f"Epoch {epoch}: No values found in ROI, skipping threshold check.")
        



######################################################           Main            ###############################################################################
train_data = dataset.train_data.feature
train_label = dataset.train_data.label

train_data_tf = to_tf(train_data)
train_label_tf = to_tf(train_label)

train_data_tf = reshape_data(train_data_tf)

# Extract every 10th element from the second axis (axis 1)

train_data_tf = train_data_tf[:, ::10, :]
train_data_tf = np.reshape(train_data_tf, (train_data_tf.shape[0], 100, 100, 1))
train_label_tf = np.reshape(train_label_tf, (train_label_tf.shape[0], 1))

#we separate train and validation data
val = int(0.25*len(train_data_tf))

train_data_tf, val_data_tf = tf.split(train_data_tf, num_or_size_splits=[len(train_data_tf)-val,val], axis=0)
train_label_tf, val_label_tf = tf.split(train_label_tf, num_or_size_splits=[len(train_label_tf)-val,val], axis=0)


epochs = 1000
tries = 25  # Maximum number of training attempts
current_try = 0  # Track number of attempts
total_epochs_trained = 0  # Track total epochs trained
threshold_value = 0.55   # Define the threshold and the name of the convolutional layer to inspect.

# Remove the first 3 images and select from the remaining ones
train_data_tf_ = train_data_tf[2:]  


m = "transformers_1"              ###########################################################################################################


model = make_model(input_shape=(100, 100))
model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mean_squared_error", 
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

# model.load_weights('transformer_model/new_models/{}/{}.weights.h5'.format(m, m))
# model = tf.keras.models.load_model('transformer_model/new_models/{}/{}.keras'.format(m, m),{"AttentionBlock":AttentionBlock})

while current_try < tries:
    # Reset model weights before each attempt
    model = reset_model()

    # Select 10 random images from the remaining dataset
    num_images = train_data_tf_.shape[0]
    sample_indices = np.random.choice(num_images, size=25, replace=False)
    sample_data = tf.gather(train_data_tf_, sample_indices)
    
    last_conv_layer_name = model.layers[3].name
    # Instantiate the callback with ten samples.


    gradcam_callback = GradCAMActivationCallback(sample=sample_data,
                                                last_conv_layer_name=last_conv_layer_name,
                                                threshold=threshold_value)

    # Train the model with the callback.
    history = model.fit(train_data_tf, train_label_tf, epochs=epochs,
                            validation_data=(val_data_tf, val_label_tf), callbacks=[gradcam_callback])

    # Check if training was stopped due to the threshold
    if gradcam_callback.stopped_due_to_threshold:
        print("Training stopped as threshold condition was met. Reinitializing weights and retrying...")
        current_try += 1  # Count this as an attempt
    else:
        print("Training completed without threshold being met.")
        break  # Exit loop if training completes normally

print(f"Training finished after {total_epochs_trained} total epochs across {current_try} tries.")         
model.save_weights('transformer_model/new_models/{}/{}.weights.h5'.format(m, m))

test_data = dataset.test_data.feature
test_label = dataset.test_data.label

test_data_tf = to_tf(test_data)
test_label_tf = to_tf(test_label)
test_data_tf = reshape_data(test_data_tf)
test_data_tf = test_data_tf[:, ::10, :]

test_label_tf = np.reshape(test_label_tf, (-1, 1))
test_data_tf = np.reshape(test_data_tf, (-1, 100, 100, 1))




with PdfPages('transformer_model/new_models/{}/Result_test.pdf'.format(m)) as pdf:
    preds, pred_table = get_prediction(test_data_tf, model)[0] # get_prediction format: ((prediction, list of predictions), )
    plt.figure(figsize= (12,1))
    plt.text(0.5, 0.5, 'RMSE of the model: {}'.format(preds , ha='center', va='center', fontsize=10))
    plt.axis('off')
    pdf.savefig()
    plt.close()    

    for i, data in enumerate(test_data_tf):
        
        fig = plt.figure(figsize=(8, 5))  # Create a new figure for each plot
        
        plt.text(0.1, 0.05, abs(pred_table.numpy()[0][i]) , ha='center', va='center', fontsize=10)
        plt.axis('off')
        
        # First subplot: display the data with imshow
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(data)  # Plot the data
        ax1.set_title(f"Qmatrix of the Cell {i+1}")  # Title for the first plot
        ax1.axis('off')  # Turn off axis for the first subplot
        cbar1 = plt.colorbar(im1, shrink=0.75, aspect=20, pad=0.05)
        rect = mplt.patches.Rectangle((50, 100), 150, 100, linewidth=2, edgecolor='red', facecolor='none')

        # Add the rectangle to the axes
        ax1.add_patch(rect)
        plt.show()
        # Generate heatmap and plot it in the second subplot
        heatmap = make_gradcam_heatmap(data, model, last_conv_layer_name)
        # with open('transformer_model/new_models/{}/heatmap_{}.pkl'.format(m, i), 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump(heatmap, f)
            
        ax2 = fig.add_subplot(122)
        resized_heatmap = zoom(heatmap, (1, 100 / 32), order=1)  # Linear interpolation
        im2 = ax2.matshow(resized_heatmap, vmin=0, vmax=1, cmap='jet')  # Display the heatmap
        cbar2 = plt.colorbar(im2, shrink=0.75, aspect=20, pad=0.05)
        ax2.set_title("Grad-CAM Heatmap")  # Title for the heatmap
        ax2.axis('off')  # Turn off axis for the second subplot

        plt.tight_layout()  # Adjust layout to prevent overlap
        pdf.savefig()
