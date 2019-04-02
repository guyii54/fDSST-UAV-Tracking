

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif

#define TIMETEST
//#define SCALE_RESULT

using namespace std;

namespace dsst
{
// Constructor
DSSTTracker::DSSTTracker(bool hog)
{
    // Parameters equal in all cases
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

}
// Initialize tracker
void DSSTTracker::init(const cv::Mat image, const cv::Rect2d &roi)
{
#ifdef TIMETEST
    double st,et,dt;
    st = clock();
#endif
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(_size_patch[0], _size_patch[1]);
    _alphaf = cv::Mat(_size_patch[0], _size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));

    train(_tmpl, 1.0); // train with initial frame

    init_scale(image, roi);
    train_scale(image, true);

#ifdef TIMETEST
    et = clock();
    dt =1000* (double)(et-st)/CLOCKS_PER_SEC;
    cout<<"init time:"<<dt<<endl;
#endif

}


// Update position and scale based on the new frame
bool DSSTTracker::update(const cv::Mat image, cv::Rect2d &roi)
{
#ifdef TIMETEST
    double st,et,dt;
    st = clock();
#endif

    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1; //let _roi.x + _roi.width = 1
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 2;
    if (_roi.width <= 0)
        _roi.width = 2;
    if (_roi.height <= 0)
        _roi.height = 2;

    //roi.x, roi.y: coordinate of top-left corner
    //cx,cy: center coordinate of roi
    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    //********************translation estimation********************
#ifdef TIMETEST
    double s_tr,e_tr,d_tr1,d_tr2;
    s_tr = clock();
#endif

	cv::Mat transFeature = getFeatures(image,0,1.0f);

#ifdef TIMETEST
    e_tr = clock();
    d_tr1 = 1000*(double)(e_tr-s_tr)/CLOCKS_PER_SEC;
    cout<<"-	-       getFeaure:"<<d_tr1<<endl;
	s_tr = clock();
#endif

    cv::Point2f res = detect(_tmpl, transFeature, _peak_value);

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale_dsst);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale_dsst);

    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 2;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 1;
    if (_roi.width <= 0)
        _roi.width = 2;
    if (_roi.height <= 0)
        _roi.height = 2;
#ifdef  TIMETEST
    e_tr = clock();
    d_tr2 = 1000*(double)(e_tr-s_tr)/CLOCKS_PER_SEC;
//	cout<<"-	-	-detect:"<<d_tr2<<endl;
    cout<<"-	tran es:"<<d_tr1+d_tr2<<endl;
#endif


    //*********************scale estimation**********************
#ifdef TIMETEST
    double s_sc,e_sc,d_sc;
    s_sc = clock();
#endif

    cv::Point2i scale_pi = detect_scale(image);
#ifdef SCALE_RESULT
    cout<<"scale result:"<<scaleFactors[scale_pi.x]<<endl;
//    cout<<"scale result:"<<scaleFactors<<endl;
#endif
    printf("dsst thresh: %f, peak: %f\n", detect_thresh_dsst, _peak_value);
    if (_peak_value >= detect_thresh_dsst)
    {
        _scale_dsst = _scale_dsst * scaleFactors[scale_pi.x];
        //printf("scale_pi.x:%d, _scale_dsst:%f\n", scale_pi.x, _scale_dsst);
        if (_scale_dsst < min_scale_factor)
            _scale_dsst = min_scale_factor;
        else if (_scale_dsst > max_scale_factor)
            _scale_dsst = max_scale_factor;

        // Compute new _roi
        cx = _roi.x + _roi.width / 2.0f;
        cy = _roi.y + _roi.height / 2.0f;
        _roi.width = base_width_dsst * _scale_dsst;
        _roi.height = base_height_dsst * _scale_dsst;
        //cout <<"rect size:"<<_roi.width<<"*"<<roi.height<<endl;
        _roi.x = cx - _roi.width / 2.0f;
        _roi.y = cy - _roi.height / 2.0f;

        if (_roi.x + _roi.width <= 0)
            _roi.x = -_roi.width + 2;
        if (_roi.y + _roi.height <= 0)
            _roi.y = -_roi.height + 2;
        if (_roi.x >= image.cols - 1)
            _roi.x = image.cols - 1;
        if (_roi.y >= image.rows - 1)
            _roi.y = image.rows - 1;
        if (_roi.width <= 0)
            _roi.width = 2;
        if (_roi.height <= 0)
            _roi.height = 2;
#ifdef TIMETEST
        e_sc = clock();
        d_sc = 1000*(double)(e_sc-s_sc)/CLOCKS_PER_SEC;
        cout<<"-	scale es:"<<d_sc<<endl;
#endif
        //********************update and train  the filter********************
#ifdef TIMETEST
        double s_fil,e_fil,d_filt1,d_filt2,d_fils;
        s_fil = clock();
#endif


        assert(_roi.width >= 0 && _roi.height >= 0);
        transFeature =getFeatures(image, 0);

#ifdef TIMETEST
        e_fil = clock();
        d_filt1 = 1000*(double)(e_fil-s_fil)/CLOCKS_PER_SEC;
        cout<<"-	-       getFeature:"<<d_filt1<<endl;
        s_fil = clock();
#endif


        train(transFeature, interp_factor);
        //attention: feature here is not the same as the feature gotten during translation estimation, _roi has changed

#ifdef TIMETEST
        e_fil = clock();
        d_filt2 = 1000*(double)(e_fil-s_fil)/CLOCKS_PER_SEC;
        cout<<"-	trans fil train:"<<d_filt1+d_filt2<<endl;
        s_fil = clock();
#endif

        train_scale(image);

#ifdef TIMETEST
        e_fil = clock();
        d_fils = 1000*(double)(e_fil-s_fil)/CLOCKS_PER_SEC;
        cout<<"-	scale fil train:"<<d_fils<<endl;
//        cout<<"-	filter train:"<<d_filt1+d_filt2+d_fils<<endl;

#endif
        roi = _roi;

#ifdef TIMETEST
    et = clock();
    dt =1000* (double)(et-st)/CLOCKS_PER_SEC;
    cout<<"update time:"<<dt<<endl;
#endif

        return true;
    }
    else
    {
        return false;
    }
}

// Detect object in the current frame.
cv::Point2f DSSTTracker::detect(cv::Mat z, cv::Mat x, float &peak_value) // KCF Algorithm 1 , _alpha updated in train();
{
    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat res = (real(dft_d(complexDotMultiplication(_alphaf, dft_d(k)), true))); // KCF (22)

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);   //&pv: max value, &pi: max location
    peak_value = (float)pv;
    trans_peak = peak_value;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols - 1)
    {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (pi.y > 0 && pi.y < res.rows - 1)
    {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image, to update _alphaf;
void DSSTTracker::train(cv::Mat x, float train_interp_factor)
{
//#ifdef TIMETEST
//    double s_train,e_train,d_train;
//    s_train = clock();
//#endif

    //trans_train1
    cv::Mat k = gaussianCorrelation(x, x);

//#ifdef TIMETEST
//    e_train = clock();
//    d_train = 1000*(double)(e_train-s_train)/CLOCKS_PER_SEC;
//    cout<<"-        -       -       -gaussioncor:"<<d_train<<endl;
//    s_train = clock();
//#endif

    //trans_train2
    cv::Mat alphaf = complexDotDivision(_prob, (dft_d(k) + lambda)); // KCF (17)


//#ifdef TIMETEST
//    e_train = clock();
//    d_train = 1000*(double)(e_train-s_train)/CLOCKS_PER_SEC;
//    cout<<"-      -       -       trans_train2:"<<d_train<<endl;
//    s_train = clock();
//#endif

    //cout<<"-    -    -sizeof k:"<<k.rows<<"*"<<k.cols<<endl;
    //cout<<"-    -    -sizeof alphaf:"<<alphaf.rows<<"*"<<alphaf.cols<<endl;

    //trains_train3
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf;


//#ifdef TIMETEST
//    e_train = clock();
//    d_train = 1000*(double)(e_train-s_train)/CLOCKS_PER_SEC;
//    cout<<"-      -       -       trans_train3:"<<d_train<<endl;
//#endif

    /*//MOSSE-style update
    cv::Mat kf = dft_d(gaussianCorrelation(x, x));
    cv::Mat num = complexDotMultiplication(kf, _prob);
    cv::Mat den = complexDotMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDotDivision(_num, _den);*/
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,
// which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).
cv::Mat DSSTTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2) // KCF (30)
{

#ifdef TIMETEST
    double s_gau,e_gau,d_gau;
    s_gau = clock();
#endif

    cv::Mat c = cv::Mat(cv::Size(_size_patch[1], _size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < _size_patch[2]; i++)
        {
            x1aux = x1.row(i); // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, _size_patch[0]);
            x2aux = x2.row(i).reshape(1, _size_patch[0]);
            cv::mulSpectrums(dft_d(x1aux), dft_d(x2aux), caux, 0, true);
            caux = dft_d(caux, true);
            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else
    {
        cv::mulSpectrums(dft_d(x1), dft_d(x2), c, 0, true); //output: c
        c = dft_d(c, true);                                //ifft
        rearrange(c);                                     //KCF page 11, Figure 6;
        c = real(c);
    }
    cv::Mat d;
    //make sure >=0 and scaling for the fft; KCF page 11
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (_size_patch[0] * _size_patch[1] * _size_patch[2]), 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);

#ifdef TIMETEST
    e_gau = clock();
    d_gau = 1000*(double)(e_gau-s_gau)/CLOCKS_PER_SEC;
    cout<<"-       -       gaussioncor:"<<d_gau<<endl;
#endif



    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat DSSTTracker::createGaussianPeak(int sizey, int sizex)
{

    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    //    printf("sizex:%d, sizey:%d\n",sizex,sizey);
    //    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    //    float mult = -(float)0.5/ (output_sigma * output_sigma);
    //
    //    output_sigma_factor = 0.1;
    //    float sigmax2 = -0.5*padding*padding/output_sigma_factor/output_sigma_factor/(sizex);
    //    float sigmay2 = -0.5*padding*padding/output_sigma_factor/output_sigma_factor/(sizey);
    //
    //
    //    for (int i = 0; i < sizey; i++)
    //        for (int j = 0; j < sizex; j++)
    //        {
    //            int ih = i - syh;
    //            int jh = j - sxh;
    //            res(i, j) = std::exp( sigmay2*ih * ih + sigmax2*jh * jh);
    //        }
    // exp(-(1/2)(sigma^2)(padding^2/(sizex*sizey)) * (x^2+y^2))
    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return dft_d(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat DSSTTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust)
{
//    cv::Rect extracted_roi;

    //printf("_roi:%f,%f,%f,%f\n",_roi.x,_roi.y,_roi.width,_roi.height);
    // get the centor of roi
    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;
    //printf("cxcy:%f,%f\n", cx, cy);
    //whether to init hann
    if (inithann)
    {

        //**********init tmpl_sz**********
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;
//        printf("padded:%d,%d",padded_w,padded_h);
        //new codes
//        _tmpl_sz.width = padded_w;
//        _tmpl_sz.height = padded_h;
//        _scale = 1;
        //此段原本为下面的代码，但根据最后输出，换成上面这段是不一样的
        //orginal code is below， codes upon have different result
        if (template_size > 1)
        {                             // Fit largest dimension to the given template size
            if (padded_w >= padded_h) //fit to width
                _scale = padded_w / (float)template_size;
            else
                _scale = padded_h / (float)template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else
        { //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }

        // Round the _tmpl_sz
        if (_hogfeatures)
        {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else
        { //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    // get new extracted_roi
//    printf("c:%f,%f,%d,%d,%f,%f,%f.%f\n", scale_adjust, _scale, _tmpl_sz.width, _tmpl_sz.height,_roi.width,_roi.height, cx, cy);
//    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;                       //无论template_size是多少，extracted_roi.width的大小均是padded_w
//    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width * _scale_dsst;                       //无论template_size是多少，extracted_roi.width的大小均是padded_w
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height*_scale_dsst;
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    if (extracted_roi.x + extracted_roi.width <= 0)
        extracted_roi.x = -extracted_roi.width + 2;
    if (extracted_roi.y + extracted_roi.height <= 0)
        extracted_roi.y = -extracted_roi.height + 2;
    if (extracted_roi.x >= image.cols - 1)
        extracted_roi.x = image.cols - 1;
    if (extracted_roi.y >= image.rows - 1)
        extracted_roi.y = image.rows - 1;
    if (extracted_roi.width <= 0)
        extracted_roi.width = 2;
    if (extracted_roi.height <= 0)
        extracted_roi.height = 2;

//    printf("extracted_roi:%d,%d,%d,%d\n", extracted_roi.x, extracted_roi.y, extracted_roi.width, extracted_roi.height);
//     cout<<"extracted_roi size:"<<extracted_roi.width<<"*"<<extracted_roi.height<<endl;
    cv::Mat FeaturesMap;
    cv::Mat z = subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height)
    {
        cv::resize(z, z, _tmpl_sz);
    }

//    printf("size of z:%d, %d \n", z.cols, z.rows);
    //double timereco = (double)cv::getTickCount();
	//float fpseco = 0;
    // HOG features
    
// #ifdef TIMETEST
//     double s_hog,e_hog,d_hog;
//     s_hog = clock();
// #endif
    if (_hogfeatures)
    {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        _size_patch[0] = map->sizeY;
        _size_patch[1] = map->sizeX;
        _size_patch[2] = map->numFeatures;

//        printf("sizeX and sizeY: %d*%d\n",map->sizeX,map->sizeY);
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map); // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();                                                                // transpose
        freeFeatureMapObject(&map);
    }
    else //CSK
    {
        FeaturesMap = getGrayImage(z);
        FeaturesMap -= (float)0.5; // CSK p10;
        _size_patch[0] = z.rows;
        _size_patch[1] = z.cols;
        _size_patch[2] = 1;
    }
// #ifdef TIMETEST
//     e_hog = clock();
//     d_hog = 1000*(double)(e_hog-s_hog)/CLOCKS_PER_SEC;
//     cout<<"-	-       getHOG:"<<d_hog<<endl;
// #endif
	//fpseco = ((double)cv::getTickCount() - timereco) / cv::getTickFrequency();
	//printf("kcf hog extra time: %f \n", fpseco);
    if (inithann)
    {
        createHanningMats();
    }
    FeaturesMap = _hann.mul(FeaturesMap); //element-wise multiplication;

//    printf("FeaturesMap size: %d*%d\n",FeaturesMap.rows,FeaturesMap.cols);

    return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame. To make the boundary of feature(image) to be zero
void DSSTTracker::createHanningMats()
{                                                                                 //Mat(size(cols,rows), type, init)
    cv::Mat hann1t = cv::Mat(cv::Size(_size_patch[1], 1), CV_32F, cv::Scalar(0)); //1 x size_patch[1]
    cv::Mat hann2t = cv::Mat(cv::Size(1, _size_patch[0]), CV_32F, cv::Scalar(0)); //size_patch[0] x 1

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t; //size_patch[0] x size_patch[1]
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        _hann = cv::Mat(cv::Size(_size_patch[0] * _size_patch[1], _size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < _size_patch[2]; i++)
        {
            for (int j = 0; j < _size_patch[0] * _size_patch[1]; j++)
            {
                _hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else
    {
        _hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float DSSTTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}



//DSST=========================================================================================
// Initialization for scales
void DSSTTracker::init_scale(const cv::Mat image, const cv::Rect2d &roi)
{
    // The initial size for adjusting
    base_width_dsst = roi.width;
    base_height_dsst = roi.height;

    // Guassian peak for scales (after fft)
    _prob_dsst = createGaussianPeak_dsst();
    _hann_dsst = createHanningMats_dsst();

    // Get all scale changing rate, DSST page 5;
    scaleFactors = new float[n_scales];
    float ceilS = std::ceil(n_scales / 2.0f);
    for (int i = 0; i < n_scales; i++)
    {
        scaleFactors[i] = std::pow(scale_step, ceilS - i - 1);
            printf("scaleFactors %d: %f; ", i, scaleFactors[i]);
    }
    printf("\n");

    // Get the scaling rate for compressing to the model size
    float scale_model_factor = 1;
    if (base_width_dsst * base_height_dsst > scale_max_area)
    {
        scale_model_factor = std::sqrt(scale_max_area / (float)(base_width_dsst * base_height_dsst));
    }
    scale_model_width = (int)(base_width_dsst * scale_model_factor);
    scale_model_height = (int)(base_height_dsst * scale_model_factor);
    //printf("%d, %d \n", scale_model_width, scale_model_height);


    // Compute min and max scaling rate
    //1.05^16 = 2.182   1.05^{-16}=0.458
    min_scale_factor = 0.01; //std::pow(scale_step,
                             //     std::ceil(std::log((std::fmax(5 / (float)base_width, 5 / (float)base_height) * (1 + scale_padding))) / 0.0086));
    max_scale_factor = 10;   //std::pow(scale_step,
                             //      std::floor(std::log(std::fmin(image.rows / (float)base_height, image.cols / (float)base_width)) / 0.0086));
    //printf("dsstInit - min_scale_factor:%f; max_scale_factor:%f;\n", min_scale_factor, max_scale_factor);
}

// Train method for scaling
void DSSTTracker::train_scale(cv::Mat image, bool ini)
{
    cv::Mat samples = get_sample_dsst(image);

    // Adjust ysf to the same size as xsf in the first frame
    if (ini)
    {
        int totalSize = samples.rows;
        _prob_dsst = cv::repeat(_prob_dsst, totalSize, 1);
    }

    // Get new GF in the paper (delta A)
    cv::Mat new_num_dsst;
    cv::mulSpectrums(_prob_dsst, samples, new_num_dsst, 0, true);

    // Get Sigma{FF} in the paper (delta B)
    cv::Mat new_den_dsst;
    cv::mulSpectrums(samples, samples, new_den_dsst, 0, true);
    cv::reduce(real(new_den_dsst), new_den_dsst, 0, CV_REDUCE_SUM);

    if (ini)
    {
        _den_dsst = new_den_dsst;
        _num_dsst = new_num_dsst;
    }
    else
    {
        // Get new A and new B, DSST (5)
        cv::addWeighted(_den_dsst, (1 - scale_lr), new_den_dsst, scale_lr, 0, _den_dsst);
        cv::addWeighted(_num_dsst, (1 - scale_lr), new_num_dsst, scale_lr, 0, _num_dsst);
    }
}

// Detect the new scaling rate
cv::Point2i DSSTTracker::detect_scale(cv::Mat image)
{

    cv::Mat samples = DSSTTracker::get_sample_dsst(image);


//    cout<<"sample size:"<<samples.rows<<"*"<<samples.cols<<endl;
//    cout<<"num size:"<<_num_dsst.rows<<"*"<<_num_dsst.cols<<endl;

    // Compute AZ in the paper
    cv::Mat add_temp;
    cv::reduce(complexDotMultiplication(_num_dsst, samples), add_temp, 0, CV_REDUCE_SUM);


//    cout<<"add_temp size:"<<add_temp.rows<<"*"<<add_temp.cols<<endl;
//    cout<<"den size:"<<_den_dsst.rows<<"*"<<_den_dsst.cols<<endl;


    // compute the final y, DSST (6);
    cv::Mat scale_response;
    cv::idft(complexDotDivisionReal(add_temp, (_den_dsst + scale_lambda)), scale_response, cv::DFT_REAL_OUTPUT);
//    cout<<"scale_response:"<<scale_response<<endl;

    // Get the max point as the final scaling rate
    cv::Point2i pi; //max location
    double pv;      //max value
    cv::minMaxLoc(scale_response, NULL, &pv, NULL, &pi);
    scale_peak = pv;
    
    return pi;
}

// Compute the F^l in DSST (4);
cv::Mat DSSTTracker::get_sample_dsst(const cv::Mat &image)
{
#ifdef TIMETEST
	double st_gsd,et_gsd,dt_gsd;
	st_gsd = clock();
#endif
	//double timereco = (double)cv::getTickCount();
	//float fpseco = 0;

    CvLSVMFeatureMapCaskade *map[n_scales]; // temporarily store FHOG result
    cv::Mat samples;                        // output
    int totalSize;                          // # of features
    // iterate for each scale
    for (int i = 0; i < n_scales; i++)
    {
        // Size of subwindow waiting to be detect
        float patch_width = base_width_dsst * scaleFactors[i] * _scale_dsst;
        float patch_height = base_height_dsst * scaleFactors[i] * _scale_dsst;

        float cx = _roi.x + _roi.width / 2.0f;          //roi.x是起点
        float cy = _roi.y + _roi.height / 2.0f;

        // Get the subwindow
        cv::Mat im_patch = extractImage(image, cx, cy, patch_width, patch_height);      //从image中提取cx,cy为中心，patch_width,patch_height为大小的图像快
        cv::Mat im_patch_resized;

        //printf("cx:%f,cy:%f\n",cx,cy);
        //printf("patch_width: %f, patch_height: %f,\n",patch_width,patch_height);
        //printf("im_patch w: %d, im_path h: %d,\n", im_patch.rows, im_patch.cols);

        if (im_patch.rows == 0 || im_patch.cols == 0)
        {
            samples = dft_d(samples, 0, 1);
            return samples;
            // map[i]->map = (float *)malloc (sizeof(float));
        }

        // Scaling the subwindow    将im_patch按scale_model_width, scale_model_height的大小resize
        //opencv中的resize函数参数：src, dst, dsize, fx, fy, flag，其中fx, fy是factor，已知放大倍数时使用，
        if (scale_model_width > im_patch.cols)
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 1);
        else
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 3);
        
        //printf("%d, %d \n", im_patch_resized.cols, im_patch_resized.rows);
        //printf("%d, %d \n", im_patch.cols, im_patch.rows);
        // Compute the FHOG features for the subwindow
        IplImage im_ipl = im_patch_resized;
        getFeatureMaps(&im_ipl, cell_size, &map[i]);
        normalizeAndTruncate(map[i], 0.2f);
        PCAFeatureMaps(map[i]);

        //赋值给sample，具体未读
        if (i == 0)
        {
            //printf("numFeatures:%d, sizeX:%d,sizeY:%d\n", map[i]->numFeatures, map[i]->sizeX, map[i]->sizeY);
            totalSize = map[i]->numFeatures * map[i]->sizeX * map[i]->sizeY;

            if (totalSize <= 0) //map[i]->sizeX or Y could be 0 if the roi is too small!!!!!!!!!!!!!!!!!!!!!!!
            {
                totalSize = 1;
            }
            samples = cv::Mat(cv::Size(n_scales, totalSize), CV_32F, float(0));
        }
        cv::Mat FeaturesMap;
        if (map[i]->map != NULL)
        {
            // Multiply the FHOG results by hanning window and copy to the output
            FeaturesMap = cv::Mat(cv::Size(1, totalSize), CV_32F, map[i]->map);
            float mul = _hann_dsst.at<float>(0, i);

            FeaturesMap = mul * FeaturesMap;
            FeaturesMap.copyTo(samples.col(i));
        }
    }
	//fpseco = ((double)cv::getTickCount() - timereco) / cv::getTickFrequency();
	//printf("kcf hog extra time: %f \n", fpseco);

    // Free the temp variables
    for (int i = 0; i < n_scales; i++)
    {
        freeFeatureMapObject(&map[i]);
    }
    // Do fft to the FHOG features row by row
    samples = dft_d(samples, 0, 1);

#ifdef TIMETEST
	et_gsd = clock();
	dt_gsd = 1000*(double)(et_gsd-st_gsd)/CLOCKS_PER_SEC;
        printf("-	-	gsd:%02f\n",dt_gsd);
#endif

    return samples;
}

// Compute the FFT Guassian Peak for scaling
cv::Mat DSSTTracker::createGaussianPeak_dsst()
{

    float scale_sigma2 = n_scales / std::sqrt(n_scales) * scale_sigma_factor;
    scale_sigma2 = scale_sigma2 * scale_sigma2;
    cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));
    float ceilS = std::ceil(n_scales / 2.0f);

    for (int i = 0; i < n_scales; i++)
    {
        res.at<float>(0, i) = std::exp(-0.5 * std::pow(i + 1 - ceilS, 2) / scale_sigma2);
    }

    return dft_d(res);
}

// Compute the hanning window for scaling
cv::Mat DSSTTracker::createHanningMats_dsst()
{
    cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
    for (int i = 0; i < hann_s.cols; i++)
        hann_s.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));

    return hann_s;
}
}
