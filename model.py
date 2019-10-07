from keras.layers import LSTM,Dense,Embedding

from keras.models import Model
from keras.layers import Input, LSTM, Dense

def build_test_model(max_source_len, max_target_len):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_source_len, ))
    encoder = LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(max_target_len-2,))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(max_target_len, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# def build_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
#     # Define an input sequence and process it.
#         encoder_inputs = Input(shape=(None, num_encoder_tokens))
#         encoder = LSTM(latent_dim, return_state=True)
#         encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#         # We discard `encoder_outputs` and only keep the states.
#         encoder_states = [state_h, state_c]
#
#         # Set up the decoder, using `encoder_states` as initial state.
#         HGE_inputs = Input(shape=(5,9,5))
#
#         HGE = LSTM(32)
#         HGE_outputs = HGE(HGE_inputs)
#
#         # We set up our decoder to return full output sequences,
#         # and to return internal states as well. We don't use the
#         # return states in the training model, but we will use them in inference.
#         decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
#         decoder_outputs, _, _ = decoder_lstm(HGE_outputs,
#                                              initial_state=encoder_states)
#         decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#         decoder_outputs = decoder_dense(decoder_outputs)
#
#         # Define the model that will turn
#         # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#         model = Model([encoder_inputs, HGE_inputs], decoder_outputs)
#
#         # Run training
#         model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#                       metrics=['accuracy'])
#         return model



