# %%
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom
from batteryml.pipeline import Pipeline, load_config, build_dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')

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
        return inputs + proj

# %%
# Create a pipeline with a configuration file, specifying the device and workspace. 
# Developers need to modify the data, feature, model and other related settings in the config file in advance. 
dataset_name = 'matr_1'
pipeline = Pipeline(config_path=f'./BatteryML/configs/baselines/nn_models/cnn/{dataset_name}.yaml',
                    workspace='workspaces')

# %%
configs = load_config(config_path=f'./BatteryML/configs/baselines/nn_models/cnn/{dataset_name}.yaml', workspace='workspaces')
dataset, raw_data = build_dataset(configs, device='cpu')

# %%
#We convert the data into tensor and extract 10% of the record that we are going to use
def to_tf(tensor):
    tensor_array = tensor.cpu().numpy()
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



# %%
train_data_tf = reshape_data(train_data_tf)

# Extract every 10th element from the second axis (axis 1)
train_data_tf = train_data_tf[:, ::10, :]
train_data_tf = np.reshape(train_data_tf, (-1, 100, 100))
train_label_tf = np.reshape(train_label_tf, (-1,1))

#we separate train and validation data
train_data_tf, val_data_tf = tf.split(train_data_tf, num_or_size_splits=[32,9], axis=0)
train_label_tf, val_label_tf = tf.split(train_label_tf, num_or_size_splits=[32,9], axis=0)

test_data = dataset.test_data.feature
test_label = dataset.test_data.label

test_data_tf = to_tf(test_data)
test_label_tf = to_tf(test_label)
test_data_tf = reshape_data(test_data_tf)
test_data_tf = test_data_tf[:, ::10, :]

test_label_tf = np.reshape(test_label_tf, (-1, 1))
test_data_tf = np.reshape(test_data_tf, (-1, 100, 100))

# %%
def make_gradcam_heatmap(img_array, model, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # Ensure the model has been built (called with input data)
    
    img_array = np.reshape(img_array, (1, 100, 100))
    grad_model = keras.models.Model(
        model.inputs, [model.layers[3].output, model.output]
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
    # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output * grads
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # heatmap = ((heatmap - tf.math.reduce_min(heatmap)) /
    #            (tf.math.reduce_max(heatmap) - tf.math.reduce_min(heatmap) + 1e-10)) * 2.0 - 1.0
    
    heatmap = tf.abs(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-16)
    
    return heatmap.numpy()

#%%
for n in range(1, 11):
    model_name = f'transformers_{n}'
    mdir = f'./transformer_models/{dataset_name}/'
    rdir = './transformer_results/'
    model = tf.keras.models.load_model(mdir+f'{model_name}/{model_name}.keras', custom_objects={'AttentionBlock':AttentionBlock})
    print(model.summary())
    
    
    prediction = model.predict(test_data_tf)
    prediction = tf.reshape(prediction, (1,len(test_data_tf)))
    rmse =  dataset.evaluate(torch.from_numpy(prediction.numpy()), 'RMSE', data_type='test')
    print(rmse)
    target = dataset.label_transformation.inverse_transform(test_label)
    prediction = dataset.label_transformation.inverse_transform(torch.from_numpy(prediction.numpy()))
    
    preds = rmse
    pred_table = (((target - prediction) ** 2) ** 0.5).numpy()[0] 
    hm = []
    for i, data in enumerate(test_data_tf):
    
        # Generate heatmap and plot it in the second subplot
        heatmap = make_gradcam_heatmap(test_data_tf[i], model)
        hm.append(heatmap)
    
        
    hm = np.array(hm)
    
    np.savez_compressed(rdir+f'{dataset_name}/res_{model_name}', heatmaps=hm, q_matrices=test_data_tf, ref=target.numpy(), pred=prediction.numpy())
    print(rmse)


