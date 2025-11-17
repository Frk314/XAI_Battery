import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import torch, pickle
import numpy as np
from scipy.ndimage import zoom
from matplotlib.backends.backend_pdf import PdfPages

import sys
from batteryml.pipeline import Pipeline, load_config, build_dataset
from batteryml.visualization.plot_helper import plot_capacity_degradation, plot_cycle_attribute, plot_result

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')



###########  Data Loading  ################
pipeline = Pipeline(config_path='configs/baselines/nn_models/cnn/matr_1.yaml',
                    workspace='workspaces')
configs = load_config(config_path='configs/baselines/nn_models/cnn/matr_1.yaml', workspace='workspaces')
dataset, raw_data = build_dataset(configs, device='cpu')
###########################


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

index = np.random.permutation(len(train_data))
train_data = train_data[index]
train_label = train_label[index]

train_data_tf = reshape_data(train_data_tf)

# Extract every 10th element from the second axis (axis 1)
train_data_tf = train_data_tf[:, ::10, :]
train_data_tf = np.reshape(train_data_tf, (train_data_tf.shape[0], 100, 100, 1))
train_label_tf = np.reshape(train_label_tf, (train_label_tf.shape[0], 1))

#we separate train and validation data
val = int(0.25*len(train_data_tf))

train_data_tf, val_data_tf = tf.split(train_data_tf, num_or_size_splits=[len(train_data_tf)-val,val], axis=0)
train_label_tf, val_label_tf = tf.split(train_label_tf, num_or_size_splits=[len(train_label_tf)-val,val], axis=0)


#learning_rate = 0.00005
batch_size = len(train_data_tf)
epochs = 1499
image_size = 100 
patch_size = 5
num_heads = 4
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
input_shape = (100, 100, 1)
num_classes = 1
mlp_head_units = [
    16,
    8 
]  # Size of the dense layers of the final classifier
#Data augmentation

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(train_data_tf)

augmented = data_augmentation(train_data_tf)

######################### CLASSES AND FUNCTIONS  ###########################

class Patches(tf.keras.layers.Layer):
    with tf.device('/CPU:0'):
        def __init__(self, patch_size):
            
            super().__init__()
            self.patch_size = patch_size
    
        def call(self, images):
            input_shape = (-1, 100, 100, 1)
            batch_size = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3]
            num_patches_h = height // self.patch_size
            num_patches_w = width // self.patch_size
            patches = tf.keras.ops.image.extract_patches(images, size=self.patch_size)
            patches = tf.keras.ops.reshape(
                patches,
                (
                    batch_size, #number of samples per patch
                    num_patches_h * num_patches_w, #number of patches
                    self.patch_size * self.patch_size * channels, #1D size of a patches
                ),
            )
            return patches
    
        def get_config(self):
            config = super().get_config()
            config.update({"patch_size": self.patch_size})
            return config


class PatchEncoder(layers.Layer):
    
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    #This function add a new dimension where the value is the position of each element in the 1D array patch
    def call(self, patch):
        positions = tf.keras.ops.expand_dims(
            tf.keras.ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    

#attention block
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, groups=1, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)# allow our class to inherit all the module of the parent class keras.layers.Layer
    
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


#Initialisation of the parameters
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def get_prediction(data, model):
    prediction = model.predict(data)
    prediction = tf.reshape(prediction, (-1,len(data)))
    return dataset.evaluate(torch.from_numpy(prediction.numpy()), 'RMSE', data_type='test')


def visualTransformer():
    with tf.device('/CPU:0'):
        inputs = keras.Input(shape=input_shape)
        augmented = data_augmentation(inputs) 
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
        # 01 attention layers
        attention_output = Attention(units=projection_dim)(encoded_patches)

        # Residual connection
        x = layers.Add()([encoded_patches, attention_output])
            
        #  normalisation of the attention result and feed forward
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = mlp(x, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.2)

        # Create a [batch_size, projection_dim] tensor.
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        
        # Add MLP.
        x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.1)
        # Classify outputs.
        outputs = layers.Dense(num_classes)(x)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


def normalize_array_01(array):
    return  (array - np.min(array)) / (np.max(array) - np.min(array))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # Ensure the model has been built (called with input data)
    
    img_array = np.reshape(img_array, (1, 100, 100))
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
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




############ WE CREATE THE PATCHES  ############################

with PdfPages('visual_transformer/patches.pdf') as pdf:
    plt.figure(figsize=(4, 4))
    image = normalize_array_01(train_data_tf[7])
    plt.imshow(image, norm=None)
    plt.axis("off")


    patches = Patches(patch_size)(image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    v_min, v_max = np.min(image), np.max(image)
    for i, patch in enumerate(patches[0][:]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.keras.ops.reshape(patch, (patch_size, patch_size, 1))
        plt.imshow(patch_img, vmin=v_min, vmax=v_max)
        plt.axis("off")
    pdf.savefig()




model = visualTransformer()

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="mean_squared_error", 
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
     
    )
    
history = model.fit(
    x=train_data_tf,
    y=train_label_tf,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_data_tf, val_label_tf)  
    ) 
    
#with PdfPages('models/history_transformer.pdf'.format(m)) as pdf:
#plt.figure(figsize= (16,5))
#plt.plot(history.history["root_mean_squared_error"], color="r")
#plt.plot(history.history['val_root_mean_squared_error'][5:], color="b")
#plt.title("Model RMSE")
#plt.xlabel("epochs")
#pdf.savefig()
    
plt.show()


m = "v_model_1"                ###########################################################################################################


test_data = dataset.test_data.feature
test_label = dataset.test_data.label

test_data_tf = to_tf(test_data)
test_label_tf = to_tf(test_label)
test_data_tf = reshape_data(test_data_tf)
test_data_tf = test_data_tf[:, ::10, :]


test_label_tf = np.reshape(test_label_tf, (test_data_tf.shape[0], 1))
test_data_tf = np.reshape(test_data_tf, (test_data_tf.shape[0], 100, 100, 1))


result = get_prediction(test_data_tf, model)
print(result)

last_conv_layer_name = model.layers[4].name

with PdfPages('visual_transformer/{}/Result.pdf'.format(m)) as pdf:
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
        resized_heatmap = zoom(heatmap, (1, 400 / 64), order=1)  # Linear interpolation
        
        # with open('visual_transformer/{}/heatmap_{}.pkl'.format(m, i), 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump(resized_heatmap, f)
        #heatmap_reshaped = tf.reshape(heatmap, (4,4))
        ax2 = fig.add_subplot(122)
        # Resize heatmap to (100, 100) using interpolation
        im2 = ax2.matshow(resized_heatmap, cmap='jet')  # Display the heatmap
        cbar2 = plt.colorbar(im2, shrink=0.75, aspect=20, pad=0.05)
        ax2.set_title("Grad-CAM Heatmap")  # Title for the heatmap
        ax2.axis('off')  # Turn off axis for the second subplot
    
        plt.tight_layout()  # Adjust layout to prevent overlap
        # pdf.savefig()
        plt.show()  # Display the figure
        #plt.pause(3)  # Pause for 3 seconds before showing the next plot



