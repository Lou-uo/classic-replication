受朋友邀请，做一个基于 Seq2Seq 模型（顺便还带上了 Transformer来练练手）的机器翻译

为了挑战自己，训练的是 ==英语→繁体== 模型。

用的环境是 [TRAE CN](https://www.trae.cn/?utm_source=360&utm_medium=360_sem&utm_campaign=48525246&utm_term=trae_sem_360_wc_pc_1_ocpc_pinp_cuopin&account_id=3524398901&ad_platform_id=360_search_lead&new_user=1&a_keywordid=70204721021&ug_ad_level_3_id=11996719315&ug_ad_level_2_id=2633808598&ug_ad_level_1_id=3015792527&ug_device=pc&ug_semver=v1.0.0&qhclickid=02d796634e9ae05e)，感觉这种基础的翻译任务，字节的 Agent IDE 就已经够用了。



数据集用的是这个⇓

https://gitcode.com/open-source-toolkit/63e8a

有一共 26000+ 条数据，用的是最原始的数据罗列投喂 🙂

```
I'd like to see a doctor.,我想看醫生。
I'd like to see a doctor.,我要看病。
I'd like you to go there.,我希望你去那裡。
I'd never done it before.,我以前從未做過它。
I'd very much like to go.,我非常想离开。
I'll be flying to Boston.,我将要飞到波士顿去。
部分数据展示
```

------

数据集处理代码 `datasets.py`

>处理数据集、分词、词汇表构建

| 模块            | 作用                                                      |
| --------------- | --------------------------------------------------------- |
| Tokenizer       | 分词器，将句子转换为索引列表                              |
| MyDataset       | PyTorch数据集类，负责加载和处理翻译数据                   |
| RegexpReplacer  | 正则表达式替换器，处理英文缩写（如 "won't" → "will not"） |
| train_val_split | 将数据集划分为训练集和验证集                              |

Tokenizer关键特性：

- 英文词典：word2index偏移+1，PAD=0
- 中文词典：word2index偏移+3，PAD=0, BOS=1, EOS=2
- encode() ：将句子转换为索引列表
- decode() ：将索引列表转换回文字

------

模型实现代码 `seq2seq.py`

>实现基于GRU和Attention机制的Seq2Seq翻译模型

| 类               | 作用                                                  |
| ---------------- | ----------------------------------------------------- |
| Encoder          | 编码器，使用Embedding和GRU将英文序列编码为隐藏状态    |
| Decoder          | 解码器基础版本                                        |
| Attention        | 注意力机制，计算上下文向量                            |
| AttentionDecoder | 带注意力机制的解码器，整合了Embedding、GRU、Attention |
| Seq2Seq          | 完整的Seq2Seq模型，整合Encoder和Decoder               |

工作流程：

1. Encoder 处理英文输入，输出encoder_output和encoder_hidden
2. AttentionDecoder 使用encoder_hidden初始化GRU hidden state
3. 逐时间步生成中文翻译，使用teacher forcing策略
4. Attention机制帮助模型关注源语言的相关部分

------

Transformer 经典代码 `transformer.py`

>实现基于自注意力机制的Transformer翻译模型

| 类                 | 作用                                                  |
| ------------------ | ----------------------------------------------------- |
| PositionalEncoding | 位置编码，使用正弦余弦函数编码位置信息                |
| MultiHeadAttention | 多头注意力机制                                        |
| PoswiseFeedForward | Position-wise前馈神经网络                             |
| EncoderLayer       | Transformer编码器层（自注意力 + 前馈）                |
| Encoder            | 完整的编码器（多层堆叠）                              |
| DecoderLayer       | Transformer解码器层（mask注意力 + 交叉注意力 + 前馈） |
| Decoder            | 完整的解码器（多层堆叠）                              |
| Transformer        | 完整的Transformer模型                                 |

核心机制：

- Self-Attention ：查询、键、值来自同一序列
- Multi-Head Attention ：多头并行处理，捕获不同子空间特征
- Positional Encoding ：注入序列位置信息
- Masked Attention ：解码器中防止看到未来信息

------

翻译交互实现代码 `translate.py`

>加载训练好的模型，进行实时翻译

核心功能：

- 加载预训练模型
- 接收用户输入的英文句子
- 自动移除标点符号
- 输出中文翻译结果

------

Seq2Seq 模型训练代码 `train_seq2seq.py`

>训练Seq2Seq模型

```python
--encoder_embedding_size 256
--decoder_embedding_size 256
--hidden_size 512
--batch-size 32
--epochs 15
--lr 1e-3
--lr_gamma 0.98
--dropout 0.3
--tp_gamma 0.99  # teacher forcing衰减率
```

训练策略：

- Teacher Forcing：训练时使用真实标签作为下一步输入
- 学习率衰减：指数衰减策略
- 梯度裁剪：防止梯度爆炸

------

Transformer模型训练代码 `train.py`

> 训练Transformer模型

```python
--encoder_embedding_size 128
--decoder_embedding_size 128
--hidden_size 256
--batch-size 32
--epochs 200
--lr 2e-4
--warmup-proportion 0.1  # 学习率预热比例
```

训练策略：

- Warmup：学习率先预热后衰减
- AdamW优化器
- Label Smoothing（如果启用）

------

**模型对比**

|    特性    |    Seq2Seq     |  Transformer   |
| :--------: | :------------: | :------------: |
|  基础架构  |   RNN (GRU)    | Self-Attention |
|   注意力   |   单一注意力   |   多头注意力   |
|   并行性   | 差（序列依赖） |       好       |
| 长距离依赖 |    难以捕获    |    容易捕获    |
|   参数量   |      较少      |      较多      |
|  训练速度  |      较慢      |  较快（相对）  |

Seq2Seq翻译流程：

```
英文句子 → Tokenizer.encode() → Encoder → Encoder Hidden
                                              ↓
中文生成 ← Decoder ← (逐词生成) ← Attention ← Encoder Output
```

Transformer翻译流程：

```
英文句子 → Tokenizer.encode() → Encoder Layers (Multi-Head Self-Attention)
                                              ↓
中文生成 ← Decoder Layers ← (逐词生成) ← Cross Attention ← Encoder Output
```

------

其实，像这种机器翻译的话，还是要用至少3090以上的卡，跑出来的效果会更好

只用 CPU 或游戏本自带的 GPU ，是完全体会不到 Transformer 的神奇之处的 😎

不过，本来只是做个小实验，就用笔记本的CPU凑合凑合了 🙃

PS：为了适配CPU，代码部分有点修改



看看初次训练后的结果：【训练1-1.png】

```cmd
请输入英文:how are you.
他們就好到了。
请输入英文:Hi.
他做了。
请输入英文:HI.,
请你找兴。
请输入英文:Begin.
开始不可以开始。
请输入英文:Hello!
帮！
请输入英文:Oh no!
不会有！
请输入英文:Got it?
懂吗？
请输入英文:He ran.
他跑得坐。
请输入英文:I konw.
我知道我们知道。
```

结果不太好，但是有那个意思了，毕竟使用笔记本跑的呢 😝

不过，应该是可以优化优化的 🧐

同时还发现个问题，就是标点符号竟然会影响翻译【训练1-2.png】

```cmd
请输入英文:Run!
不是真的，我們跑步。
请输入英文:Run?
汤姆在火车暗了吗？
请输入英文:Run
顯急時地方便跑步。
请输入英文:Run.
跑。
```

初步猜测是Transformer的attention把标点也计算上了



修改策略如下：

1. 修改了 Teacher Forcing 策略 ： tp_gamma=0.99 ，让 teacher forcing 概率随训练逐渐减少

2. 增强了模型容量 ：

   - 编码器/解码器嵌入维度：256（从128）

   - 隐藏层大小：512（从256）

3. 调整了学习率策略 ：

   - 初始学习率：1e-3（从2e-3）

   - 学习率衰减：0.98（从0.99）

4. 增加了训练轮数 ：200（从100）

5. 增强了正则化 ：dropout=0.3（从0.1）

6. 在`translate.py`中添加标点符号处理功能
   - 导入了 `string` 模块
   - 在翻译前使用 `str.maketrans` 和 `translate` 方法移除所有标点符号
   - 使用清理后的文本进行翻译

其实，我为了省时间，只训练了13个epoch，不过就是那个意思，时间问题而已。

再加上数据集比较小，效果可能欠佳。

第二次训练的结果【训练2.png】

```cmd
请输入英文:how are you
你“到你真好。
请输入英文:I love you
轮爱你我爱愛你。
请输入英文:Who are you
誰选你给你们谁。
请输入英文:Go!
從週六掛號碼去露出去。
请输入英文:Run?
顯急時地方便跑步。
请输入英文:Wow
哇麻錶明天才華。
请输入英文:Torrow
明天测试着明白。
请输入英文:He is Tom
千Tom 他是湯姆的一間头？
请输入英文:I go to school tomorrow   
明天我去学校学校结婚了。
请输入英文:What would you like to eat tonight?
你想吃甚麼東西吃。
请输入英文:Can you use this programming tool?
抓住你能留下来。
请输入英文:The completion of this task is not satisfactory.
這事是有中的說的機會。
```

现在的缺陷主要是在训练设备和训练数据上了（好像一开始就是这两的问题），代码肯定是没问题的 🥰

然后加上字符串处理后，也不会受到标点符号的影响了

那就暂时先告一段落了 🥳