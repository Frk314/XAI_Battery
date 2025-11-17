import tensorflow as tf
import keras
from keras import layers, models, Input


def make_model(input_shape=(100, 100, 1), num_classes=1):
    # Input layer
    inputs = Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(16, 3, padding="valid")(inputs)  # 3x1 kernel
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling2D(2)(x)

    # Intermediate blocks
    
    x = layers.Conv2D(8, 3, padding="valid")(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling2D(2)(x)

    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4, activation="relu")(x)

    # Output layer
    outputs = layers.Dense(num_classes)(x)

    # Create the model using functional API
    model = models.Model(inputs, outputs)

    return model


# model = make_model(input_shape=(100, 100, 1))
# model.summary()

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


def make_transformer_model(input_shape=(100, 100), num_classes=1):
    # Input layer
    inputs = Input(shape=input_shape)

    # Entry block
    x, s = TransformerEncoder(32)(inputs)

    # Output layer
    outputs = layers.Dense(num_classes)(x)

    # Create the model using functional API
    model = models.Model(inputs, outputs)

    return model


# model = make_model(input_shape=(100, 100))
# model.summary()