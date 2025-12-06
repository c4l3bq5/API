import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
from huggingface_hub import hf_hub_download
from config import Config

@register_keras_serializable(package="Custom")
class CustomReLU(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.relu(inputs)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="Custom")
class PReLUParam(layers.Layer):
    def __init__(self, init_alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha
        self.alpha = None

    def build(self, input_shape):
        if self.alpha is None:
            self.alpha = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.Constant(self.init_alpha),
                trainable=True,
                name='alpha_pr',
                dtype=self.dtype
            )
        super().build(input_shape)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = self.alpha * (inputs - tf.abs(inputs)) * 0.5
        return pos + neg

    def get_config(self):
        config = super().get_config()
        config.update({'init_alpha': self.init_alpha})
        return config


@register_keras_serializable(package="Custom")
class MaxPoolWithArgmax(layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='SAME', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.padding = padding.upper()

    def call(self, inputs):
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=self.padding,
            include_batch_in_index=False
        )
        return output, tf.cast(argmax, tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class ModelLoader:
    def __init__(self):
        self.custom_objects = {
            'CustomReLU': CustomReLU,
            'PReLUParam': PReLUParam,
            'MaxPoolWithArgmax': MaxPoolWithArgmax
        }
        self.model_ternary = None
        self.model_upper = None
        self.model_lower = None

    def load_models(self):
        print("Descargando modelos desde HuggingFace...")

        try:
            print("  [1/3] Ternary Classifier...")
            path_ternary = hf_hub_download(
                repo_id=Config.HF_MODEL_TERNARY,
                filename=Config.HF_FILE_TERNARY
            )
            self.model_ternary = keras.models.load_model(path_ternary, custom_objects=self.custom_objects)
            print("  OK Ternary")

            print("  [2/3] Upper SegNet...")
            path_upper = hf_hub_download(
                repo_id=Config.HF_MODEL_UPPER,
                filename=Config.HF_FILE_UPPER
            )
            self.model_upper = keras.models.load_model(path_upper, custom_objects=self.custom_objects)
            print("  OK Upper")

            print("  [3/3] Lower SegNet...")
            path_lower = hf_hub_download(
                repo_id=Config.HF_MODEL_LOWER,
                filename=Config.HF_FILE_LOWER
            )
            self.model_lower = keras.models.load_model(path_lower, custom_objects=self.custom_objects)
            print("  OK Lower")

            print("Todos los modelos cargados")

        except Exception as e:
            print(f"ERROR: {e}")
            raise

        return self

    def get_models(self):
        return self.model_ternary, self.model_upper, self.model_lower


_model_loader_instance = None

def get_model_loader():
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader().load_models()
    return _model_loader_instance