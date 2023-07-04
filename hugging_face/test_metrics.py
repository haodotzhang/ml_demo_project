from datasets import list_metrics

#列出评价指标
metrics_list = list_metrics()
a, b = len(metrics_list), metrics_list
print(a)
print(b)


from datasets import load_metric

#加载一个评价指标
# metric = load_metric('glue', 'mrpc')   # MRPC(The Microsoft Research Paraphrase Corpus，微软研究院释义语料库) # 加载不了，网络问题
# print(metric.inputs_description)

#计算一个评价指标
metric = load_metric('accuracy', 'f1')
predictions = [0, 1, 0]
references = [0, 1, 1]
final_score = metric.compute(predictions=predictions, references=references)
print(final_score)