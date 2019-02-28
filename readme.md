# 毕业设计代码

## 改动kcftracker.cpp与kcftracker.hpp
删除了一些函数和多分支（LAB，FIXEDWINDOW等）的赋值，更改了namespace为kcf，程序运行流畅，帧数约为13FPS（70ms）

## 参数表
|参数名|含义|值|
|:-:|:-:|:-:|
scale_step = 1.05  
scale_weight = 0.95	//to downweight detection scores of other   scales for added stability  
template_size = 96   
scale_padding = 1.0	//extra area surrounding the target for scaling  
scale_sigma_factor = 0.25	//bandwidth of Gaussion  
n_scales = 33;  
scale_lr = 0.025;	//scale learning rate  
scale_max_area = 512;	//max ROI size before compressing  
scale_lambda = 0.01;	//regularization  
_alphaf;//alpha in paper, use this to calculate the detect result, changed in train();  
_prob;  //Gaussian Peak(training outputs);  
_tmpl;  //features of image (or the normalized gray image itself  when raw), changed in train();  
_num;   //numerator: use to update as MOSSE  
_den;   //denumerator: use to update as MOSSE   

## 核心函数
getFeature()  
creatGaussianPeak()  
tran()  -> tran_tranlation  
init_dsst()	-> init_scale  
train_dsst  ->  train_scale  

## 训练
训练分两种方式：MOSSE方式和KCF的方式，还没比较两种方法的速度，可能KCF更快，因为KCF是MOSSE后面提出的弄懂了KCF的原理提出的更新方法。
- KCF方式将滤波器看成一个整体进行更新，即更新_alphaf，OpenTracker项目中用的均是这种方法
- MOSSE方式分为分子和分母进行更新，即更新_num, _den，DSST论文中用的是这种方式
### KCF的更新方法
更新滤波器(_alpha)
$$k^{xx} = gaussanCorrelation(x,x)$$
$$\alpha = \frac{\hat{y}}{dft\_d(k^{xx})+\lambda}$$
$$\alpha = (1 - train\_interp\_factor)  \alpha + (train\_interp\_factor)\alpha$$
更新模板


