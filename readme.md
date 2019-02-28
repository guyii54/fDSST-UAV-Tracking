# 毕业设计代码

## 改动kcftracker.cpp与kcftracker.hpp
删除了一些函数和多分支（LAB，FIXEDWINDOW等）的赋值，更改了namespace为kcf，程序运行流畅，帧数约为13FPS（70ms）

## Parameters
|参数名|含义|值|
|:-:|:-:|:-:|
translation:  
detect_thresh_kcf = 0.13;		//dont know meaning  
template\_size = 96   			//template size in pixels, 0 to use ROI size,the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers  
lambda = 0.0001;				//regularization  
padding = 1.5;					//area surrounding the target, relative to its size  
out_put_sigma_factor = 0.125;	//bandwidth of gaussian target  

if(hog)/else  
interp_factor= 0.012/0.075;		//linear interpolation factor for adaptation  
sigma = 0.6/0,2;				//gaussian kernel bandwidth  
cell\_size = 4/1; 				//hog cell size if(hog)->cell_size=4 else->cell_size=1  


scale:  
scale_step = 1.05;				//scale step for multi-scale estimation, 1 to disable it, a in DSST article  
scale_weight = 0.95;			//downweight detection scores of other   scales for added stability  
scale_padding = 1.0				//extra area surrounding the target for scaling  
scale_sigma_factor = 0.25		//bandwidth of Gaussion  
n_scales = 33;  				//number of scales
scale_lr = 0.025;				//scale learning rate  
scale_max_area = 512;			//max ROI size before compressing  
scale_lambda = 0.01;			//regularization  


## Variables
_roi			//input in init, output in update;  
_alphaf			//alpha in paper, use this to calculate the detect result, changed in train();  
_prob	  		//Gaussian Peak(training outputs);  
_tmpl	  		//features of image (or the normalized gray image itself  when raw), changed in train();  
_num	   		//numerator: use to update as MOSSE  
_den	   		//denumerator: use to update as MOSSE  
_size_patch		//0:rows;1:cols;2:numFeatures; init in getFeatures();  

scale:  
scaleFactors	//a^n in DSST article

## Functions
init();  
translation init + scale init.  
translation init:  
1. init _size_patch and get features  
2. use feature create \hat{y}, also named _prob
3. use KCF(17) to train filter _alphaf  
scale init(init_scale+ tran_scale):  
1. init_scale():get scaleFactors  
2. tran_scale():get samples using scaleFactors and use them to train the filter  

update();  
1. detect translation and scale using function detect() and detect_scale()
2. update the filter of translation and scale using train() and train_scale()

train();  
1. train filter _alphaf
	1. get_kxx
<p align="center">
    <img src="equation/get_kxx.png"> 
</p>
	2. get filter this frame
<p align="center">
    <img src="equation/get filter.png"> 
</p>
	3. update filter
<p align="center">
    <img src="equation/update_filter.png"> 
</p>
2. update feature _tmpl
<p align="center">
    <img src="equation/update_tmpl.png"> 
</p>


train_scale();  


## 核心函数
getFeature()  
creatGaussianPeak()  
init()
tran()  
init_dsst()	-> init_scale  
train_dsst  ->  train_scale  

## 训练
训练分两种方式：MOSSE方式和KCF的方式，还没比较两种方法的速度，可能KCF更快，因为KCF是MOSSE后面提出的弄懂了KCF的原理提出的更新方法。
- KCF方式将滤波器看成一个整体进行更新，即更新_alphaf，OpenTracker项目中用的均是这种方法
- MOSSE方式分为分子和分母进行更新，即更新_num, _den，DSST论文中用的是这种方式
### KCF的更新方法(funtion train())
更新滤波器(_alpha)  
step1 get_kxx
<p align="center">
    <img src="equation/get_kxx.png"> 
</p>
step2 get filter this frame
<p align="center">
    <img src="equation/get filter.png"> 
</p>
step3 update filter
<p align="center">
    <img src="equation/update_filter.png"> 
</p>
更新模板


