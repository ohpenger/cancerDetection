### Segmentation Model
##### A experiment project from book \<\<Deep Learning With Pytorch\>\> aims to train a classification model to detect suspect tumor.
##### This segmentation model is going to identify suspect nodule candidates from raw CT scans data
##### dataset: luna datasets which contains 888 CT scans with high-precision manual annotation which were collected during a two-phase annotation process using 4 experienced radiologists
##### model: Unet(2d) remark: The 3d data would be sliced and feed into 2d Unet
##### optimizer: Adam
##### loss function: dice loss
##### performance: got  0.0332 precision, 0.8397 recall and 0.0639 f1 score in validation dataset


### Classification Model
##### A experiment project from book \<\<Deep Learning With Pytorch\>\> aims to train a classification model to detect suspect tumor.
##### This classification model will attempt to tell a nodule candidate we get from segmentation part from nodule or non-nodule.
##### dataset: suspect nodule candidates provided by segmentation part.
##### model: custom 3d model
##### optimizer: SGD
##### loss function: crossEntropy
##### performance: got  0.9964 precision, 0.9974 recall and 0.9969 f1 score in validation dataset