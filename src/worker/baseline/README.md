# 说明

baseline采用glm4-8b-chat模型，直接进行lora微调，去掉lm head，后面接入一个预设的drug layer层做分类

验证集分布：721条数据
=== 标签数量分布 ===
平均标签数: 6.15
最少标签数: 1
最多标签数: 17
=== Top 20 药物分布 ===
阿托伐他汀钙片: 261
阿司匹林肠溶片: 209
阿卡波糖: 178
二甲双胍: 126
硝苯地平控释片: 120
氨氯地平片: 103
甘精胰岛素: 98
碳酸钙片: 93
甲钴胺: 91
雷贝拉唑: 81
门冬胰岛素: 80
依帕司他: 78
瑞巴派特片: 74
氯吡格雷: 69
美托洛尔缓释片: 68
兰索拉唑: 66
左甲状腺素钠片: 59
银杏叶软胶囊: 59
阿法骨化醇: 57
瑞舒伐他汀: 54
=== token 分布 ===
avg token count: 1531.2649098474342
max token count: 5844
min token count: 396

训练集分布：2881样本
avg token count: 1531.64352655328
max token count: 8095
min token count: 379

=== 标签数量分布 ===
平均标签数: 6.06
最少标签数: 1
最多标签数: 21

=== Top 20 药物分布 ===
阿托伐他汀钙片: 1037
阿司匹林肠溶片: 782
阿卡波糖: 706
二甲双胍: 483
碳酸钙片: 424
甘精胰岛素: 402
甲钴胺: 379
氨氯地平片: 375
硝苯地平控释片: 359
门冬胰岛素: 301
瑞巴派特片: 300
依帕司他: 295
雷贝拉唑: 288
美托洛尔缓释片: 270
阿法骨化醇: 249
瑞舒伐他汀: 242
兰索拉唑: 241
氯吡格雷: 217
银杏叶软胶囊: 208
左甲状腺素钠片: 194

baseline model精度如下：
==================================================
Test set evaluation results:
==================================================
jaccard_th0.3: 0.1261
precision_avg_th0.3: 0.1323
recall_avg_th0.3: 0.7024
f1_avg_th0.3: 0.2156
score_th0.3: 0.1709
f1_micro_th0.3: 0.2196
precision_micro_th0.3: 0.1297
recall_micro_th0.3: 0.7155
jaccard_th0.4: 0.2050
precision_avg_th0.4: 0.2514
recall_avg_th0.4: 0.5069
f1_avg_th0.4: 0.3148
score_th0.4: 0.2599
f1_micro_th0.4: 0.3345
precision_micro_th0.4: 0.2473
recall_micro_th0.4: 0.5168
jaccard_th0.5: 0.2054
precision_avg_th0.5: 0.3617
recall_avg_th0.5: 0.3018
f1_avg_th0.5: 0.3015
score_th0.5: 0.2535
f1_micro_th0.5: 0.3447
precision_micro_th0.5: 0.3900
recall_micro_th0.5: 0.3088
avg_labels_per_sample: 6.1484
avg_preds_per_sample: 4.8682
exact_match_accuracy: 0.0222
total_samples: 721