import random

def make_vocab(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	for doc in docs:
		for w in doc:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
	return w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
	ps = []
	while len(ps) < batch_size:
		ps.append(random.randint(0, len(docs_source) - 1))

	source_batch = []
	target_batch = []

	source_lens = [len(docs_source[p]) for p in ps]
	target_lens = [len(docs_target[p]) + 1 for p in ps]

	max_source_len = max(source_lens)
	max_target_len = max(target_lens)

	for p in ps:
		source_seq = [w2i_source[w] for w in docs_source[p]] + [w2i_source["_PAD"]] * (
					max_source_len - len(docs_source[p]))
		target_seq = [w2i_target[w] for w in docs_target[p]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]] * (
					max_target_len - 1 - len(docs_target[p]))
		source_batch.append(source_seq)
		target_batch.append(target_seq)

	return source_batch, source_lens, target_batch, target_lens