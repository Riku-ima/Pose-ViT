# Pose-ViT

ViTにpose estimataionを合わせる。
現状 Densenet , AcT , Timesformerをベースにモデルを比較中

## 現状
既存:  AcT  Timesformer
作成:　AcT+Timesformer 
Test:  embeddingをTubletに
精度が40%付近 。そのままだとやはりCNN baseに負けるので、大規模データで事前学習を行う。

将来的には Tubletとpose baseをviewにしてmulti view化

## 参考文献(コード)
[1]Facebook(2021)
https://github.com/facebookresearch/TimeSformer 

[2]hobin , (2021)
https://github.com/hkair/Basketball-Action-Recognition
