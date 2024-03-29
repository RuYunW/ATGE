# from keras.layers import LSTM,Dense,Embedding
from keras import Model
from keras.layers import Input, LSTM, Dense, Dropout

def build_test_model(max_source_len, node_num, max_col, code_length):
    encoder_inputs = Input(shape=(max_source_len, 1))
    encoder_outputs, state_h, state_c = LSTM(code_length, return_state=True)(encoder_inputs)

    encoder_state = [state_h, state_c]
    input_onehot = Input(shape=(max_col, code_length))
    x = LSTM(code_length, activation='relu', return_sequences=True)(input_onehot)
    x = LSTM(code_length, activation='softmax', return_sequences=False)(x, initial_state=encoder_state)
    x = Dense(code_length)(x)
    x = Dropout(0.2)(x)
    x = Dense(code_length)(x)
    x = Dropout(0.2)(x)
    output_onehot = Dense(code_length, activation='softmax')(x)


    model = Model([encoder_inputs, input_onehot], output_onehot)
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



