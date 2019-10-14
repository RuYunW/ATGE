from json2txt_1 import load_dataset, fileDev
from analyze_2 import conclude_signature
from io_list_3 import generate_io
from node_onehot_generate6 import get_node_onehot
from cited2citing_4 import cited2citing, write_cited
from content_5 import content_file_generation

# Deviding the original dataset into program & describe two parts
dataset = load_dataset('../data/test.json')  # list
program,describe = fileDev(dataset)

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



