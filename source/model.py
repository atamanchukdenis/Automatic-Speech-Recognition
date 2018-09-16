import keras.backend as tffrom keras import Modelfrom keras.initializers import RandomNormalfrom keras.layers import Input, Dense, CuDNNLSTM, Bidirectional, ReLU, BatchNormalization, Dropoutdef get_model(input_dim=26, fc_sizes=[128, 128, 64], fc_dropouts=[0, .1, 0],              rnn_sizes=[512, 512, 512], output_dim=36, random_seed=4567,              stddev=0.046875):    input_tensor = Input([None, input_dim], name='X')    random = RandomNormal(stddev=stddev, seed=random_seed)    x = input_tensor    for fc_size, fc_dropout in list(zip(fc_sizes, fc_dropouts)):        linear = __get_fc_layer(fc_size, init=random, activation='linear')        x = linear(x)        x = ReLU(max_value=20)(x)        if fc_dropout:            x = Dropout(fc_dropout)(x)    for rnn_size in rnn_sizes:        rnn = __get_rnn_layer(rnn_size)        x = rnn(x)        # Does not help (high overfit) and has to be applied along all layers        # (also fc layers). Bigger batch_size required, so the training has        # fewer gradient updates. The data is limited...        # x = BatchNormalizationWrapper(rnn_size)(x)    softmax = __get_fc_layer(output_dim, init=random, activation='softmax')    output_tensor = softmax(x)    model = Model(input_tensor, output_tensor, name='DeepSpeech')    return modeldef __get_fc_layer(fc_size, init, activation):    return Dense(units=fc_size, kernel_initializer=init,                 use_bias=False, activation=activation)def __get_rnn_layer(rnn_size):    return Bidirectional(        CuDNNLSTM(rnn_size, kernel_initializer='glorot_uniform',                  return_sequences=True, return_state=False),        merge_mode='sum')class BatchNormalizationWrapper(BatchNormalization):    """    Sequence-wise Batch Normalization. Do not use Lambda layers due to    serialization problems (load/save model).    """    def __init__(self, layer_size, **kwargs):        self.layer_size = layer_size        super().__init__(**kwargs)    def call(self, x_source):        x = tf.reshape(x_source, [-1, self.layer_size])        x = super().call(x)        x = tf.reshape(x, tf.shape(x_source))        return x