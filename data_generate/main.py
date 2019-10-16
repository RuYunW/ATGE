from data_generate.data_generate_utils import *

# Deviding the original dataset into program & describe two parts
dataset = load_dataset('../data/test.json')  # list
program, describe = fileDev(dataset)
write(describe, '../data/describe.txt')

# Conclude the Signature information from program
method = conclude_signature(program)

# make the method into three parts
inlist, outlist, method_list = generate_io(method)


# using iolist generate node ont-hot code
node_onehot_code = get_node_onehot(inlist, outlist)

# generate the node info into cited2citing(cites) file
cc = cited2citing(inlist, outlist, method_list)
write_cited(cc, '../data/cited.txt')

# generate the node info into content file
content_file_generation(node_onehot_code)



