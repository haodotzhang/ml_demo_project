from transformers import BertTokenizer

#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
    cache_dir='./model_dir',  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    force_download=False,   
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

print(tokenizer)
print(sents)


#编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],  # 一次编码两个句子，若没有text_pair这个参数，就一次编码一个句子

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',   # 少于max_length时就padding
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,  # None表示不指定数据类型，默认返回list
)

print(out)

tokenizer.decode(out)


#增强的编码函数
out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tensorflow,pytorch,numpy,默认值None为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

print(out)   # 字典
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'])


#批量编码一个一个的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],  # 批量编码，一次编码了两个句子(与增强的编码函数相比，就此处不同)

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=15,
    add_special_tokens=True,
    
    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

print(out)   # 字典
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1])


#批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],   # tuple

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

print(out)
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'][0])


#获取字典
zidian = tokenizer.get_vocab()

a, b, c = type(zidian), len(zidian), '月光' in zidian,   # (dict, 21128, False)
print(a, b, c)


#添加新词
tokenizer.add_tokens(new_tokens=['月光', '希望'])

#添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})   # End Of Sentence

zidian = tokenizer.get_vocab()

a, b, c, d = type(zidian), len(zidian), zidian['月光'], zidian['[EOS]']   # (dict, 21131, 21128, 21130)
print(a, b, c, d)


#编码新添加的词
out = tokenizer.encode(
    text='月光的新希望[EOS]',
    text_pair=None,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    
    return_tensors=None,
)

print(out)

tokenizer.decode(out)