from model import build_model
from utils import *

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra-eng/fra.txt'

docs_source = [['where', 'are', 'you', 'come', 'from'],
               ['good', 'morning', 'how', 'are', 'you'],
               ['nice', 'to', 'meet', 'you', 'too'],
               ['today', 'is', 'a', 'wonderful', 'day'],
               ['cats', 'are', 'somewhat', 'like', 'dogs']]

docs_target = [['a', 'p', 'p', 'l', 'e'],
               ['jin', 'wan', 'ying', 'gai', 'bu', 'ao', 'ye'],
               ['lv', 'chen', 'is', 'so', 'handsome'],
               ['wo', 'xiang', 'chi', 'pao', 'mian', 'hao', 'e', 'a'],
               ['shi', 'zai', 'bian', 'bu', 'chu', 'lai', 'le']]

docs_graph = [[[0,0,0,1,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1]],
              [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]],
              [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,1],],
              [[0,0,0,0,1],[0,0,0,1,0],[1,0,0,0,0],],
              [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,1,0],]]





w2i_source, i2w_source = make_vocab(docs_source)
w2i_target, i2w_target = make_vocab(docs_target)

source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target, w2i_target,
                                                                 batch_size)
num_encoder_tokens = 5
num_decoder_tokens = 9

encoder_input_data = source_batch
HGE_input_data = docs_graph
decoder_target_data = target_batch

model = build_model(num_encoder_tokens,num_decoder_tokens,latent_dim)

model.fit([encoder_input_data, HGE_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')



