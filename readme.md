# 毕业设计代码

## 改动kcftracker.cpp与kcftracker.hpp
删除了一些函数和多分支（LAB，FIXEDWINDOW等）的赋值，更改了namespace为kcf，程序运行流畅，帧数约为13FPS（70ms）

## Parameters
|参数名						|值					|含义|
|:-:|:-:|:-|
|translation:|				|					|  |
|detect\_thresh_kcf			|0.13				|dont how to set|
|template\_size				|96					|template size in pixels, 0 to use ROI size,the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers|
|lambda						|0.0001				|regularization| 
|padding					|1.5				|area surrounding the target, relative to its size|
|out_put_sigma_factor		|0.125				|bandwidth of gaussian target|
|if(hog)/else:|				|					| 
|interp_factor				|0.012/0.075		|linear interpolation factor for adaptation|
|sigma						|0.6/0.2			|gaussian kernel bandwidth|  
|cell_size					|4/1				|hog cell size|
|							|					|
|scale:						|					|
|n_scales					|33					|number of scales|
|scale_step					|1.05				|a in DSST article |
|scale_weight				|0.95				|downweight detection scores of other   scales for added stability|
|scale_padding				|1.0				|extra area surrounding the target for scaling|  
|scale_sigma_factor			|0.25				|bandwidth of Gaussion|
|scale_lr					|0.025				|scale learning rate| 
|scale_max_area				|512				|max ROI size before compressing|
|scale_lambda				|0.01				|regularization|


## Variables
|变量名					|解释|
|:-|:-|
|全局变量|				|		|
|_roi					|input in init, output in update;  |
|_alphaf				|位置滤波器|
|_prob					|理想输出，在init中创建，从未改变，用来训练位置滤波器|
|_tmpl					|先验知识模板,每一帧用它与本帧的特征做匹配，结束后用这一帧的特征更新模板|
|_num					|numerator: use to update as MOSSE,size与samples的相同  |
|_den					|denumerator: use to update as MOSSE，size为1*n_scale  |
|_size_patch			|0:rows;1:cols;2:numFeatures; init in getFeatures();  |
|scaleFactors			|a^n in DSST article|
|_scale_dsst			|当前帧的尺度相对第一帧roi的参数，即当前预测的尺度绝对值就是_scale_dsst*base_width_dsst|
|base_width_dsst		|第一帧图像给的roi的尺寸|
|scale_model_width		|尺度预测中，所有尺度的图像块均要resize到这一尺寸后再进行hog特征的提取|
|samples				|尺度预测提取的样本集，size为x*n_scale|




## Functions
---

### init();  
translation init + scale init.  
translation init:  
1. **getFeature()**函数初始化_size_patch???，提取特征feature并初始化特征模板_tmpl
2. 用_size_patch创建_prob
3. 用_tmpl训练 **train()** 位置滤波器_alphaf

scale init(init_scale+ tran_scale):  
1. init_scale():初始化n_scale个尺度的值，储存在scaleFactors中
2. tran_scale():用scalesFactors提样本sample，并用它们训练尺度滤波器_num_dsst与_den_dsst

---

### update();  
1. 用detect()和detect_scale()函数检测出位置与尺度
2. 用train()和train_scale()函数更新位置滤波器和尺度滤波器

---

### train();  
#### 1.训练滤波器_alphaf
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

#### 2.训练模板_tmpl
<p align="center">
    <img src="equation/update_tmpl.png"> 
</p>

---

### train_scale();  
#### 1.获得特征samples
#### 2.用samples训练尺度滤波器的分子_den_dsst与分母_num_dsst
step1 get filter this frame  
<p align="center">
    <img src="equation/get A.png"> 
</p>
<p align="center">
    <img src="equation/get B.png"> 
</p>
step2 update filter  
<p align="center">
    <img src="equation/update A.png"> 
</p>
- train_scale()的输入是原图，而train()的输入是特征

---

### detect();
#### 输入：
z = _tmpl  			训练好的特征模板
x = getFeature(img)	当前帧图像块的特征
#### 输出：
peak_value			峰值
res 				预测的位置
#### 1.得到相关分数
相关分数：论文中的f(z)，代码中的res  
step1 得到kxz  
step2 复数点除complexDotDivision得到res  
<p align="center">
    <img src="equation/get res.png"> 
</p>
step3 最大响应赋值给peak_value

#### 2.得到分数最大位置pi、最终预测位置p
step1 minMaxLoc()函数寻找pi  
step2 subPixelPeak()  改变pi至p   


---

### get_sample_dsst
获得多尺度尺度的特征
#### 1.得到待提取图像块的尺寸
```
float patch_width = base_width_dsst * scaleFactors[i] * _scale_dsst;
float patch_height = base_height_dsst * scaleFactors[i] * _scale_dsst;
```
#### 2.在原图中提取图像块
```
cv::Mat im_patch = extractImage(image, cx, cy, patch_width, patch_height); 
```
cx,cy均为本帧预测出的图像中心

#### 3.对图像块进行resize
统一resize到scale_model_width,scale_model_height的大小  
scale_model_width,scale_model_height的来源：  
init_scale()函数中初始化，即该变量一经初始化后是不再变化的，大小为base_width_dsst x scale_model_factor,即：  
**若第一帧的图像块面积不大于max_area，那么以后提取的尺度图像块大小均与第一帧图像块的大小一样**

---

## 训练
训练分两种方式：MOSSE方式和KCF的方式，还没比较两种方法的速度，可能KCF更快，因为KCF是MOSSE后面提出的弄懂了KCF的原理提出的更新方法。
- KCF方式将滤波器看成一个整体进行更新，即更新_alphaf，OpenTracker项目中用的均是这种方法
- MOSSE方式分为分子和分母进行更新，即更新_num, _den，DSST论文中用的是这种方式

## 亚像素极值点检测 subpixel peak detection
代码中的subpixelpeak()函数  
参考博客[https://www.cnblogs.com/shine-lee/p/9419388.html](https://www.cnblogs.com/shine-lee/p/9419388.html)  
此处使用抛物线近似  
<p align="center">
    <img src="equation/subpixelpeak 1.png"> 
</p>
<p align="center">
    <img src="equation/subpixelpeak 2.png"> 
</p>
<p align="center">
    <img src="equation/subpixelpeak 3.png"> 
</p>


## 公式推导
为毕业设计成文方便，见公式推导.docx

## 关于编码格式
编码格式会影响opencv的解码速度，进而影响程序的整体运行速度  
cap>>frame 这行代码运行时间长  
经测试，H.264是较慢的编码格式，但MJEP的编码格式视频质量太差  
想办法加速解码速度？


## 测试结果
---

### PC各段时间测试
测试视频 201903062.MP4	h.264编码,n_sacle = 33  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|23			|
|解码			|						|4			|
|init			|						|3.404		|
|update			|						|19.3		|
|update			|tranlation estimation	|3			|
|update			|scale estimation		|6.5		|
|update			|filter train			|9.6		|

---

### n_scale与scale_step的修改
---
#### 速度测试  
测试视频 201903062.MP4	h.264编码,n_sacle = 9  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|12.679		|
|解码			|						|4			|
|init			|						|\			|
|update			|						|8.533		|
|update			|tranlation estimation	|3.337		|
|update			|scale estimation		|1.46		|
|update			|filter train			|4.64		|

测试视频 201903062.MP4	h.264编码,n_sacle = 20  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|15.2		|
|解码			|						|4			|
|init			|						|\			|
|update			|						|11.54		|
|update			|tranlation estimation	|2.68		|
|update			|scale estimation		|5.7		|
|update			|filter train			|6.7		|

测试视频 landing19030722.MP4	h.264编码,n_sacle = 15 scale_step = 1.1  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|36.4	|
|解码			|						|4			|
|init			|						|\			|
|update			|						|32.2		|
|update			|tranlation estimation	|3.68		|
|update			|scale estimation		|16.2		|
|update			|filter train			|12.25		|



#### 调参事项
1. 最终是要找到一条很好的尺度变化曲线  
2. 当尺度过小时，如8时，尺度结果始终是1，即没有响应，当15时开始有响应  
3. 论文中给出的n_scale是33，scale_step是1.05  
4. 速度是和第一帧的roi大小相关的，因为后面提取的图像块都要resize到base_scale上再提hog特征  

---
















