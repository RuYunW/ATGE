from keras.models import Model
from keras.layers import Input, LSTM, Dense


def build_model(num_encoder_tokens,num_decoder_tokens,latent_dim):
    # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        HGE_inputs = Input(shape=(5,9,5))

        HGE = LSTM(32)
        HGE_outputs = HGE(HGE_inputs)

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(HGE_outputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, HGE_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


        #
        # # Next: inference mode (sampling).
        # # Here's the drill:
        # # 1) encode input and retrieve initial decoder state
        # # 2) run one step of decoder with this initial state
        # # and a "start of sequence" token as target.
        # # Output will be the next target token
        # # 3) Repeat with the current target token and current states
        #
        # # Define sampling models
        # encoder_model = Model(encoder_inputs, encoder_states)
        #
        # decoder_state_input_h = Input(shape=(latent_dim,))
        # decoder_state_input_c = Input(shape=(latent_dim,))
        # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # decoder_outputs, state_h, state_c = decoder_lstm(
        #     decoder_inputs, initial_state=decoder_states_inputs)
        # decoder_states = [state_h, state_c]
        # decoder_outputs = decoder_dense(decoder_outputs)
        # decoder_model = Model(
        #     [decoder_inputs] + decoder_states_inputs,
        #     [decoder_outputs] + decoder_states)
        #
        # # Reverse-lookup token index to decode sequences back to
        # # something readable.
        # reverse_input_char_index = dict(
        #     (i, char) for char, i in input_token_index.items())
        # reverse_target_char_index = dict(
        #     (i, char) for char, i in target_token_index.items())