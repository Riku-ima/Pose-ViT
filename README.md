# Pose-ViT

ViTにpose estimataionを合わせる。
現状 Densenet , AcT , Timesformerをベースにモデルを比較中

## 現状
既存:AcT  Timesformer

作成:AcT+Timesformer 

Test:embeddingをTubletに

他に参考になりそうなTransformerモデルの作成。
精度が40%付近 。そのままだとCNN baseに負けているので、大規模データで事前学習を行う。

将来的には [Multiview Transformers for Video Recognition](https://arxiv.org/pdf/2201.04288.pdf)  を参考にTubletとpose baseをviewにしてmulti view化

![image](https://user-images.githubusercontent.com/61176769/223036006-cb85fd8a-6538-4917-81ed-fdad8f9badb4.png)
このようにし、テスト

## 参考文献(コード)
[1]Facebook(2021)
https://github.com/facebookresearch/TimeSformer 

[2]hobin , (2021)
https://github.com/hkair/Basketball-Action-Recognition
