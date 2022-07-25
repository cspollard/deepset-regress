# modified from Dan:
#
# https://github.com/dguest/flow-network/blob/8acc708469ab45ee221d46fbd036f61de20fc2a5/SumLayer.py#L4-L29


from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Avg(Layer):
    """
    Averaging layer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:,:,None]
        return K.average(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
