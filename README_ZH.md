## 基于词袋模型(Bow)的场景图像分类系统————以15-Scene数据集为例

---
记一次计算机视觉实验作业
### 主要任务：
使用传统视觉方法（视觉词袋模型）完成图像分类

### 主要思路：
提取sift局部特征，进行KMeans聚类以获取视觉词汇(visual words)，使用每个图像包含的视觉词汇的统计直方图特征训练SVM。

### 数据集划分
每个类别取前150个图像作为训练集，剩余的用于验证

### 超参数设置：
1. 增加特征提取器的`nOctiveLayers`有助于提取更丰富的局部特征表示，但代价是计算量也提升了，实验中取值为5
2. KMeans聚类簇数`n_clusters`，也就是词汇表的大小，其数量越大可以提取更加细粒度和丰富的图像表示，也有助于提高准确率
3. SVM模型的kernel，使用rbf径向基函数效果最佳
4. SVM模型的C值：这里取10.0，也还不错

以上模型参数下，模型准确率可以达到60%

### 文档

- 实验报告文档见(文档)[https://github.com/hammershock/VisionVocabClassifier/tree/main/docs]

- 参考论文见(参考论文)[http://people.csail.mit.edu/torralba/courses/6.870/papers/cvpr06b.pdf]
