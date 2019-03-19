## task9

1. 基本的Attention原理。参考翻译任务中的attention。

2. HAN的原理（Hierarchical Attention Networks）。

3. 利用Attention模型进行文本分类。

   参考：https://blog.csdn.net/qq_24305433/article/details/80427159

   参考：https://github.com/magical2world/tf-hierarchical-rnn





### Attention：

![1553001788705](C:\Users\tadsh\AppData\Roaming\Typora\typora-user-images\1553001788705.png)

![1553001829339](C:\Users\tadsh\AppData\Roaming\Typora\typora-user-images\1553001829339.png)

![1553002172792](C:\Users\tadsh\AppData\Roaming\Typora\typora-user-images\1553002172792.png)





### HAN结构：

![1553002391209](C:\Users\tadsh\AppData\Roaming\Typora\typora-user-images\1553002391209.png)

整个网络结构包括四个部分：

　　1）词序列编码器

　　2）基于词级的注意力层

　　3）句子编码器

　　4）基于句子级的注意力层

　　整个网络结构由双向GRU网络和注意力机制组合而成，具体的网络结构公式如下：

　　1）词序列编码器

　　　　给定一个句子中的单词 witwit ，其中 ii 表示第 ii 个句子，tt 表示第 tt 个词。通过一个词嵌入矩阵 WeWe 将单词转换成向量表示，具体如下所示：

　　　　　　xit=We;witxit=We;wit 

　　　　接下来看看利用双向GRU实现的整个编码流程：

　　　　![img](https://img2018.cnblogs.com/blog/1335117/201809/1335117-20180926194344302-660535312.png)

　　　　最终的 hit=[→hit,←hit]hit=[→hit,←hit] 。

　　2）词级的注意力层

　　　　注意力层的具体流程如下：

![1553002456501](C:\Users\tadsh\AppData\Roaming\Typora\typora-user-images\1553002456501.png)

　　　　上面式子中，uituit 是 hithit 的隐层表示，aitait 是经 softmaxsoftmax 函数处理后的归一化权重系数，uwuw 是一个随机初始化的向量，之后会作为模型的参数一起被训练，sisi 就是我们得到的第 ii 个句子的向量表示。

 　　3）句子编码器

　　　　也是基于双向GRU实现编码的，其流程如下，

　　　　![img](https://img2018.cnblogs.com/blog/1335117/201809/1335117-20180926195416649-1757826136.png)

　　　　公式和词编码类似，最后的 hihi 也是通过拼接得到的

　　4）句子级注意力层

　　　　注意力层的流程如下，和词级的一致

　　　　![img](https://img2018.cnblogs.com/blog/1335117/201809/1335117-20180926195555464-266318749.png)

　　　　最后得到的向量 vv 就是文档的向量表示，这是文档的高层表示。接下来就可以用可以用这个向量表示作为文档的特征。

**3、分类**

　　直接用 softmaxsoftmax 函数进行多分类即可

　　　　![img](https://img2018.cnblogs.com/blog/1335117/201809/1335117-20180926195849825-1952907260.png)

　　损失函数如下：

　　　　![img](https://img2018.cnblogs.com/blog/1335117/201809/1335117-20180926195937154-923631599.png)



代码，跑的imdb数据，batchsize=5，隐藏层的节点数为128

结果：

​	Epoch 0, test accuracy is 0.859532