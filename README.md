# Registration_CNN_NTG

## Paper
- "A Multispectral Image Registration Method Based on Unsupervised Learning"(基于无监督学习的多光谱图像配准)

## Poster
For the speed of the web loading, the image of the poster has been put in the link(poster-compressed-public.jpg)

## Train
The average training progress costs about 6 hours.
- Train.py: The geometric-model we used is the affine model and the training data is generatered online through random affine params.
- model/cnn_registration_model.py: the main model of our model, which contains feature extraction, feature matching, feature regression.
- ntg_pytroch/register_loss.py: This file contains our unsupervised loss function, which is first proposed by [this paper](https://www.researchgate.net/publication/321231034_Normalized_Total_Gradient_A_New_Measure_for_Multispectral_Image_Registration).

## Test
- multispectral_pytorch_batch.py: We use two-stage registeration progress to achieve the sub-pixel level accuracy. Firstly, the deep model is used to estimate the rough affine params. Then we will use the traditional ntg method to optimize the rough params.

## Visualization
For the purpose of visualization, we add the pyqt client to use our method quickly.

## FAQ
If you have other questions, welcone to submit issues.
