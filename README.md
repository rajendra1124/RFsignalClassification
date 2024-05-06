# RFsignalClassification
Dataset with Label of NR, LTE, NR_LTE.
 
Each dataset contains around 900 images.

rf1_with_confusionMatrix.py is ResNet model code for classification.

rfSigInVitPlotConMatrix.py is ViT model python code for classification.

Matlab file is used to extract feature of wireless signal using conventional method. Using 2D FFT, time, frequency, phase, modulation, bandwidth, sub-carrier spacing and SSB features are calculated, however the features extracted are not same as used to generate the signal. There is deviation between used features to generate signal and the detected features. 
