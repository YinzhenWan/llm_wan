from transformers import AutoModelForCausalLM,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/Users/wanyinzhen/PycharmProjects/llm_wan/tokenizer')
zh_demo = '床前明月光，疑是地上霜。举头望明月，低头思故乡。'
print(tokenizer.tokenize(zh_demo))
print(tokenizer.encode(zh_demo))

'''
['åºĬ', 'åīį', 'æĺİæľĪ', 'åħī', 'ï¼Į', 'çĸĳ', 'æĺ¯', 'åľ°ä¸Ĭ', 'éľľ', 'ãĢĤ', 'ä¸¾', 'å¤´', 'æľĽ', 'æĺİæľĪ', 'ï¼Į', 'ä½İå¤´', 'æĢĿ', 'æķħä¹¡', 'ãĢĤ']
[2693, 559, 29962, 1013, 249, 2725, 299, 9807, 12776, 256, 1391, 1116, 1432, 29962, 249, 39922, 1414, 20327, 256]
'''
en_demo = 'what can I say? Mamba out!'
print(tokenizer.tokenize(en_demo))
print(tokenizer.encode(en_demo))
'''
['wh', 'at', 'Ġcan', 'ĠI', 'Ġsay', '?', 'ĠM', 'amba', 'Ġout', '!']
[6662, 297, 2655, 539, 18606, 37, 437, 40618, 2159, 7]
'''

code_demo = 'import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport torch\n'
print(tokenizer.tokenize(code_demo))
print(tokenizer.encode(code_demo))
'''
['im', 'port', 'Ġnum', 'py', 'Ġas', 'Ġn', 'p', '\n', 'im', 'port', 'Ġmat', 'pl', 'ot', 'l', 'ib', '.', 'py', 'pl', 'ot', 'Ġas', 'Ġpl', 't', '\n', 'im', 'port', 'Ġpand', 'as', 'Ġas', 'Ġp', 'd', '\n', 'im', 'port', 'Ġtor', 'ch', '\n']
[586, 1525, 2810, 42627, 640, 544, 86, 60929, 586, 1525, 5378, 1737, 550, 82, 1522, 20, 42627, 1737, 550, 640, 962, 90, 60929, 586, 1525, 21377, 347, 640, 350, 74, 60929, 586, 1525, 22572, 600, 60929]
'''

# 测试一下训练好的分词器