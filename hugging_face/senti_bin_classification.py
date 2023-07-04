'''
（一）定义数据集
用的还是 ChnSentiCorp数据集，是一个中文情感分析数据集，包含酒店、笔记本电脑和书籍的网购评论
'''
# 加载数据集
import torch
from datasets import load_from_disk

#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        # 加载数据。ChnSentiCorp为消费评价数据集，分好评和差评
        self.dataset = load_from_disk('./data/ChnSentiCorp')[split]  # Available splits: ['test', 'train', 'validation']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label

dataset = Dataset('train')
a, b = len(dataset), dataset[0]  # 训练集有9600句话；dataset[0]表示第1句话，前面是text，后面是label，1表示差评
print("a: ", a)
print("b: ", b)


'''
(二) 加载tokenizer，即字典和分词工具
这里使用的分词工具是bert-base-chinese（要跟预训练模型相匹配）
'''
#加载字典和分词工具，即tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
    cache_dir='./model_dir',  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    force_download=False,   
)
print(tokenizer)


'''
（三）定义批处理函数
在批处理函数中要做分词和编码，并取出分词之后的结果
'''
# 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,   # 当句子长度大于max_length时，截断
                                   padding='max_length',   # 一律补0到max_length长度
                                   max_length=500,
                                   return_tensors='pt',   # 返回pytorch类型的tensor
                                   return_length=True)   # 返回length，标识长度

    input_ids = data['input_ids']    # input_ids:编码之后的数字
    attention_mask = data['attention_mask']     # attention_mask:补零的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']   # 第一个句子和特殊符号的位置是0，第二个句子的位置是1(包括第二个句子后的[SEP])
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


'''
（四）定义数据加载器，查看数据样例
'''
#数据加载器
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

# 查看数据样例
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(data_loader):
    break

print(len(data_loader))  # 600 = 9600/16
a, b, c, d = input_ids.shape, attention_mask.shape, token_type_ids.shape, labels   # 500表示句子最大长度为500
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)


'''
（五）加载bert中文模型
'''
from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese') # 这个入参列表还需要研究下
print(pretrained)

'''
这里不使用fine-tuning，直接把预训练模型的参数冻结住，只训练它的下游任务模型，对预训练模型本身的参数不做调整
'''
#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)  # 这里不使用fine-tuning，直接把预训练模型的参数冻结住，只训练下游任务模型，对预训练模型本身的参数不调整

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)

out_shape = out.last_hidden_state.shape   # [batch_size, 数据分词的长度(每一句话编码成500个词的长度), 词编码的维度(即每一个词编码成一个768维的向量)]
print("out_shape: ", out_shape)


'''
（六）定义下游任务模型
下游任务模型是一个单层网络模型，只包括一个全连接神经网络。计算过程：

预训练模型抽取数据特征，得到的维度是 [batch_size, 768]
把抽取出来的特征放到全连接网络中去运算，得到的维度是 [batch_size, 2]
'''
#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)  # 单层网络模型，只包括了一个fc的神经网络

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,   # 先拿预训练模型来做一个计算，抽取数据当中的特征
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        # 把抽取出来的特征放到全连接网络中运算，且特征的结果只需要第0个词的特征(跟bert模型的设计方式有关。对句子的情感分类，只需要拿特征中的第0个词来进行分类就可以了)
        out = self.fc(out.last_hidden_state[:, 0])   # torch.Size([16, 768]) -> [16, 2]
        
        # 将softmax函数应用于一个n维输入张量，对其进行缩放，使n维输出张量的元素位于[0,1]范围内，总和为1
        out = out.softmax(dim=1)  

        return out

model = Model()
print("cls_model: ", model)

cls_out = model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape    # torch.Size([16, 2])
print("cls_out: ", cls_out)


'''
（七）训练下游任务模型
训练过程中用到的优化器是AdamW, 计算CrossEntropyLoss 并梯度下降即可
'''
from transformers import AdamW   # 优化器，即 Adam + Weight decay(自适应梯度方法)

#训练下游任务模型
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()  # Pytorch计算交叉熵误差的函数自带softmax，故训练时模型里不要添加softmax

model.train()

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(data_loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)  # 输出跟真实的labels计算loss
    loss.backward()   # 调用反向传播得到每个要更新参数的梯度
    optimizer.step()  # 每个参数根据上一步得到的梯度进行优化
    optimizer.zero_grad()  # 把上一步训练的每个参数的梯度清零

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print(i, loss.item(), accuracy)

    if i == 10:  # 只训练300个轮次，没有把全量的数据(len(data_loader)= 600 = 9600/16)全部训练一遍
        break


'''
（八）测试
'''
#测试
def test():
    model.eval()
    
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break   # 只测试前5个

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)  # 测试集的正确率

test()