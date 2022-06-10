# Locally enhanced Convolutional Transformer(LeCT)

Since the introduction of Vision Transformer (ViT), the discussion on which model is better among ViT, CNN and MLP models has continued. However, the discussion has not been settled and research continues.
In this situation, what I felt while looking at the studies on several pure models is that each model has different strengths, and the model we propose uses the strengths of each model appropriately.
In particular, our model properly blends the strengths of CNN and ViT. In this report, we propose a model called Locally enhanced Convolutional Transformer (LeCT). There are three main strengths of the Locally enhanced Attention module used in this model that distinguish it from the existing Attention module.
1) Calculates the projection to query, key, and value “Sequentially” using convolution. Through this, attention can be performed through a locally reinforced representation.
2) Utilizes Long-range features and local features simultaneously through convolutional skip connection and multi-head self-attention using local features.
3) has linear computational complexity with respect to sequence length by utilizing inductive bias that correlation between adjacent pixels on 2D image is high
The results in the limited experimental environment show that our model performs better than the latest models.

See below for details.
[Report(Written in Korean)](https://github.com/ysj9909/Vision_Backbone_projects/blob/main/LeCT_report.pdf)

## Model

### Architecture
![Model_Architecture](https://user-images.githubusercontent.com/93501772/172985941-8b1d5e88-49e9-4022-bf89-62ff8f4c5145.png)

### Locally enhanced Attention
![LeAtt](https://user-images.githubusercontent.com/93501772/172986110-3221a967-64a3-4c29-bc41-ce14a044bd50.png)


#### Results
![Results](https://user-images.githubusercontent.com/93501772/172986299-e51d6887-6035-46d5-ae73-d7338bf978ad.png)


----
The baselines code is heaviliy based on official code.
[ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py), 
[CeiT](https://github.com/coeusguo/ceit/blob/main/ceit_model.py), 
[AlterNet](https://github.com/xxxnell/how-do-vits-work/blob/transformer/models/alternet.py)
