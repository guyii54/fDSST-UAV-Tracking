# 毕业设计代码

## 改动kcftracker.cpp与kcftracker.hpp
删除了一些函数和多分支（LAB，FIXEDWINDOW等）的赋值，更改了namespace为dsst，程序运行流畅，720P帧数约为13FPS（70ms）,帧数会依据第一帧初始化的框的大小而有较大改变

## Parameters
|参数名						|值					|含义|
|:-:|:-:|:-|
|translation:|				|					|  			|
|detect\_thresh_kcf			|0.13				|dont how to set|
|template\_size				|96					|template size in pixels, 0 to use ROI size,the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers|
|lambda						|0.0001				|regularization| 
|padding					|1.5				|area surrounding the target, relative to its size|
|out_put_sigma_factor		|0.125				|bandwidth of gaussian target|
|							|					|			|
|if(hog)/else:|				|					| 			|
|interp_factor				|0.012/0.075		|linear interpolation factor for adaptation|
|sigma						|0.6/0.2			|gaussian kernel bandwidth|  
|cell_size					|4/1				|hog cell size|
|							|					|			|	
|scale:						|					|			|
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

### get_sample_dsst()
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

### detect_scale()
预测尺度
#### 1.得到samples
```
cv::Mat samples = DSSTTracker::get_sample_dsst(image);
```
samples的尺度是w*n_scale, w的含义未知

#### 2.求得分矩阵分子
```
cv::reduce(complexDotMultiplication(_num_dsst, samples), add_temp, 0, CV_REDUCE_SUM);
```
reduce是将一个矩阵变成一个向量，即最后的add_temp尺寸为1*15, 与分母_den_dsst的尺寸相同

#### 3.求得分矩阵
```
cv::idft(complexDotDivisionReal(add_temp, (_den_dsst + scale_lambda)), scale_response, cv::DFT_REAL_OUTPUT);
```
scale_response尺寸也为1*n_scale，即每个尺寸一个得分，求最高的

---

### getFeature()
位置预测特征提取  
#### 1. 中心点
中心点是_roi的中心点  
```
float cx = _roi.x + _roi.width / 2;
float cy = _roi.y + _roi.height / 2;
```

#### 2. 特征区域大小
**KCF中的区域大小extracted_roi没有变化过，大小均是上一帧的roi大小乘padded**   
即
```
padded_w = _roi.width * padding;
extracted_roi.width = padded_w;
```

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


## 关于HOG特征提取部分代码的阅读
该部分共三个函数（fhog.cpp）：
1. getFeatureMaps();  
2. normalizeAndTruncate();  
3. PCAFeatureMaps();  

---

### getFeatureMaps()函数
变量表：  
|变量名				|变量含义							|
|：-					|：-									|
|k(cell_size)		|cell size							|
|sizeX				|待提取图横向有多少个cell				|
|sizeY				|待提取图纵向有多少个cell				|
|width				|待提取图的宽						|
|height				|待提取图的高						|
|numChannels		|待提取图的通道数					|
|NUM_SECTOR			|nbins 方向数						|
|map				|提取出的特征谱						|
|kernel_dx			|水平方向微分核						|
|kernel_dy			|垂直方向微分核						|
|arg_vector			|离散的方向角  i*pi/nbins			|
|boundary x			|cos(arg_vector)					|
|boundary y			|sin(arg_vector)					|
|dx					|横向的梯度结果 size:width*height*numChannels	|
|dy					|纵向的梯度结果						|
|datadx				|指向图第j行的微分值地址的指针		|
|x,y				|第i行，第j列，第c通道的横纵向梯度值	|
|r					|存梯度最大的通道的梯度绝对值 size:width*height	|  

函数步骤：  
step1 声明变量、存储空间
声明以上变量，声明map的存储空间  

step2 滑动核求梯度
使用filter2D函数求得横纵向梯度dx,dy  

step3 计算梯度绝对值
计算三个通道的梯度值  
```
r = squar(x*x+y*y);  
```
取最大的梯度最大的存到r中  

step4 计算梯度方向  
用离散化的方向角计算  
```
dotProd = boundary_x[i]*x+boundary_y[i]*y;
```
取最大的arg_vector[i]为方向角  

step5 赋值给map  


---

### allocFeatureMapObject（）函数
声明特征谱(map)的空间，并将map全部赋值为0   
函数原型 int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, 
                          const int sizeY, const int numFeatures)  
obj：结构体的名字
sizeX、sizeY、numFeatures:要声明的obj的参数
 
---


## 关于编码格式
编码格式会影响opencv的解码速度，进而影响程序的整体运行速度  
cap>>frame 这行代码运行时间长  
经测试，H.264是较慢的编码格式，但MJEP的编码格式视频质量太差  
想办法加速解码速度？

## 关于是否需要更新模板（interp_factor）
代码中关于模板更新的部分：  
step1. init函数中初始化_tmpl  
step2. train函数中用这一帧的特征更新_tmpl  
若一直更新模板，那么跟踪算法用到的就不仅仅是第一帧目标的特征，若跟踪过程中出现一段较长时间的错误，则模板完全错误，无法再回到原目标，比如当视频中目标尺度略微变化，而跟踪框认为上一帧的尺度分数更高，那么这一帧的尺度就会更新入模板中，多有这样的几帧，算法所认为的尺度就完全发生变化了，就出现了再也跟不准尺度的现象。  
若不更新模板，那么跟踪算法从头到尾只使用第一帧框出来的目标特征，而本算法中提取的是HOG特征，因此若不更新模板，对旋转的情况是完全不具有适应性的。  
结论：在本场景中，保持一个较小的interp_factor是较为简单的解决方法  

## 关于参数template_size的重要性
原理性问题未知，只说现象：   
1. template用在getFeature函数中。在位置滤波器进行初始化init时，对所提取的extracted_roi大小进行微调，虽然extracted_roi的大小仅有几个像素的变化，但是微调后的extracted_roi在提取特征时速度非常快。这样也是因为template_size的值的选取是有讲究的，一般取2的指数。越小速度越快但是容易跟不准，越大速度越慢但效果更好。这里取了128   
2. 第一帧初始化后，位置滤波器提取的extracted_roi大小不再变化。   


## 关于位置滤波器的特征提取步骤
step1. 根据上一帧的roi大小(_scale_dsst * base_taget_sz)、padding得到这一帧的特征提取区域（extracted roi），这一区域尺寸是变化的  
step2. extracted roi resize到一个固定的尺寸（_tmpl_sz），固定统一尺寸便于做相关运算，便于更新。  
step3. 做相关运算，得到_tmpl_sz这个尺度上的响应最大位置坐标（res）  
step4. 根据resize的映射关系，将res映射回原图的尺度中
```
real-shift = resized-shift * cell_size * _scale_dsst
```


## 关于n_scale和scale_step的选择


## TLD算法
各部分任务：  
tracker: 针对目标的运动进行预测（利用帧间关系）  
detector: 将每一帧视为独立的进行全图的检测，会存储目前未知所有的正负模板样本  
learning: 观察tracker和detector，预测检测器的错误，产生样本去避免这些错误  

detector:  
全图检测方法：用窗口滑动滑出约50k个图像块，用cascaded方法逐步筛选出有目标的图像块  
cascad方法：分成多个stage，一个stage是一个阈值条件，剔除一部分没有目标的图像块  
stage1 Patch variance  
灰度变化低于50的留下，其余的剔除，也叫variance filter  
stage2 Ensemble clssifier  
n个滤波器，每个滤波器产生一个code，对目标进行评价  
stage3 Nearest neighbor classifier  

## STAPLE算法
staple算法在DSST基础上加入了CN这一新的特征，即利用了颜色信息，分为template和histogram两个评价体系来预测位置，这个方法对于形变物体具有较好的适应性  


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

测试视频 201903062.MP4	h.264编码,n_sacle = 20, PC  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|15.2		|
|解码			|						|4			|
|init			|						|\			|
|update			|						|11.54		|
|update			|tranlation estimation	|2.68		|
|update			|scale estimation		|5.7		|
|update			|filter train			|6.7		|

测试视频 landing19030722.MP4	h.264编码,n_sacle = 15 scale_step = 1.1, PC   

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|36.4		|
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

### manifold test

测试视频 DJI_0144.MP4	n_sacle = 15  MPED4, MANIFOLD  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|62			|
|解码			|						|10			|
|init			|						|29			|
|update			|						|50			|
|update			|tranlation estimation	|20			|
|update			|scale estimation		|6			|
|update			|filter train			|24			|


测试视频 scale1.MP4	n_sacle = 15  h.264, MANIFOLD, init window 119*114  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|117		|
|解码			|						|61			|
|init			|						|\			|
|update			|						|56			|
|update			|tranlation estimation	|22			|
|update			|scale estimation		|6			|
|update			|filter train			|27.7		|

测试视频 landing2.MP4	n_sacle = 15  h.264, MANIFOLD, init window 132*118  

|主函数			|子函数					|时长(ms)	|
|:-				|:-						|:-			|
|总程序			|						|117		|
|解码			|						|61			|
|init			|						|\			|
|update			|						|97			|
|update			|tranlation estimation	|35			|
|update			|scale estimation		|6			|
|update			|filter train			|46			|



## 待拍摄（测试）的数据

## 日志
3.18.    
视频末端被跟踪物体变大后位置滤波器很容易出问题，出现漂移的问题  
原因：位置滤波器提取特征的大小只与第一帧大小有关，以后不再变大，第一帧只有70*70像素的目标最后变成了300*300的大小，位置滤波器还是以120*120的大小在提特征  
3.19.   
继续测试视频，对3.18.的问题寻找解决方法。  
目前的思考是，虽然特征框不再变大，但由于靶标是内嵌的，因此内部也是有HOG信息的，对于靶标这种跟踪物体来说，位置滤波器的大小不变是没有问题的，即目标变大，位置滤波器提取的特征就是内部的HOG特征。只需要稍稍增大学习率(0.012->0.1)，让模板更新稍稍变快即可。因此现在在对学习率进行调整。但是问题还没有得到解决。  
对于舰船这种目标，可能内部特征不是很好，没有什么轮廓、纹理。  
另一种思路是让extracted_roi大小跟着变化，但是提取尺寸的问题还没得到很好的解决  
晚上解决了18号的问题，在阅读了原论文matlab版代码后，发现matlab代码在位置滤波器提取特征时使用的extracted_roi大小：  
```
patch_sz = floor(model_sz * currentScaleFactor);
... ...
... ...(subwindow img and get im_patch)
im_patch = resize(im_patch, model_sz);
```
即先用currentScaleFactor（本代码中的_scale_dsst）乘model_sz(=base_target_sz*padding)得到本帧提取的区域大小（变化的），提取子区域后，在resize到统一的model_sz（不变的）大小，用resize后的图像提取hog特征。再用这个特征去做相关得到最后的响应，取响应最大处得到位置预测结果,再用resize的关系将resize后的最大响应坐标映射回原图中：  
```
_roi.x = _roi.x + res.x * cell_size * _scale_dsst;
_roi.y = _roi.y + res.y * cell_size * _scale_dsst;		//realshift = resizedshift * cell_size * _scale_dsst
```

另外在测试DJI_0144视频的时候，尺度滤波器没有跟上尺度的变化，因此稍稍减小了scale filter的lr(0.025->0.02)，减小了scale_step(1.05->1.03)  

3.20.  
目前计划：1.测试视频，继续找bug；2. 优化尺度曲线；3. 优化参数    

3.25.  
讨论了毕业设计要做的工作  
1.近处跟踪靶标的软件已基本完成，对尺度、位移适应性较强  
2.需要完成的新功能，在远处识别到舰船后，用视觉跟踪舰船，这一部分就需要对旋转亦有适应性，因此需要使用新的算法，对于新算法的速度问题，解决方法有二：使用GPU；硬件平台使用TX2或manifold2  

3.27.  
调研了目前的目标跟踪算法对于旋转的适应性  
旋转的适应性分为两种：非平面的旋转（out-of-plain rotation）和平面内的旋转  
- 平面内的旋转，目前有些方法可以解决，ECO算法可以解决，另一种叫LDES(AAAI 2019)的算法也可以同时尺度和平面内旋转的问题，它将图像放在对数极坐标(log-polar)，在极坐标中尺度和旋转就变成了普通的位移，这种在对数极坐标中的方法亦称为傅立叶梅林方法  
- 平面外的旋转，目前ECO称对平面外的旋转具有适应性，但论文中给出的例子也不是很理想  

在TX2上对目前的fDSST程序进行了测试，速度并没有明显的提升，随后对程序的各个部分进行了详细的测试  
目前耗时较长的几个模块：  
1. getFeature();
	大小为247*232的图像块，耗时14ms
2. get_sample_dsst();
	大小为247*232的图像块，耗时17ms
3. gaussianCorrelation();
	大小为196*185的图像块，耗时8ms  

归结起来（主耗时程序）：  
translation estimation = getFeature()+detect()  
detect() = gaussioncorrelation()  
scale estimation = get_sample_dsst()  
translation train = getFeature()+gaussioncorrelation()  
scale estimation = get_sample_dsst()  



3.28.  
打算用gpu加速hog特征的提取  
阅读了fDSST中HOG特征的提取过程  
阅读了opencv sample中HOG特征提取的例程  
还没有依照例程实现gpu提取HOG特征  


3.29.  
使用opencv中封装好的HOG特征提取的函数，报错，没有运行成功，返回学习cuda编程基础，想看懂代码  


4.2.  
无GPU，有了效果较好的一版  
```
	detect_thresh_kcf = 0.13;
    detect_thresh_dsst = 0.15;
    lambda = 0.0001;
    padding = 1.5;
    output_sigma_factor = 0.125; //0.1

    if (hog)
    { // HOG - KCF
        // VOT
        interp_factor = 0.1;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5;
        cell_size = 4; //hog cell size = 4;
        _hogfeatures = true;
    }
    else
    { // RAW - CSK
//        interp_factor = 0.075;
        interp_factor = 0.01;
        sigma = 0.2;
        cell_size = 1; //just pixel;
        _hogfeatures = false;
    }

     //multiscale=========
        template_size = 128; // 64 or 96 is  a little small;
        scale_weight = 0.95;

    //dsst===================
        scale_step = 1.03;
        n_scales = 15;
//        _dsst = true;
        _scale_dsst = 1;
        scale_padding = 1.0;
        scale_sigma_factor = 0.25;
        scale_lr = 0.02;
        scale_max_area = 512;
        scale_lambda = 0.01;

```
4.2.测试视频小结  
使用DJI无人机拍摄的视频进行测试  
1.帧率  
视频分辨率640*360  
程序帧数可稳定在20fps以上  
2.各项适应性  
- 快速移动：具有较好的适应性(car8)  
- 尺度变化：具有较好的适应性(boat1)  
- 小目标(20*30)：跟踪不稳定，跟踪框有些抖动(boat6)  
- 旋转：不具有适应性(boat5)  
- 会车、会船：若跟踪框内背景较少，可适应，若跟踪框内包括了其他船，无法适应(boat5,car3)  
- 遮挡：不具有适应性(car4)  

4.4.  
使用opencv cuda库提取hog特征：  
出现以下问题：  
1.目标图片过小时，程序报错  
2.hog特征如何转化成程序中可用的特征图（feature map）未知  
feature map使用的fhog特征，一方面size较小，另一方面提取较快

4.6.  
TLD论文  
STAPLE论文  
SSE及NEON：SSE是英特尔开发的单指令多数据流指令集（SIMD），类似于并行处理的原理让它可以对一些算法进行加速，neon和sse类似，是属于ARM Cortex A系列的扩展结构。

4.7，  
找到了staple的代码，运行测试后发现比只用DSST效果好很多：
- 能基本解决非平面旋转的问题
- 能解决短时间遮挡的问题
- 适用neon加速提取fhog特征，720P速度在20帧左右
- 尺度部分偶尔会逐渐放大，需要debug