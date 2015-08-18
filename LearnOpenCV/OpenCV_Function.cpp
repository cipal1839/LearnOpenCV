#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include <iostream>  
#include "opencv/cv.h"
#include "OpenCV_Function.h"

#define WINDOW_NAME1 "【原始图片】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【效果窗口】"        //为窗口标题定义的宏 

int g_nMatchMethod;
int g_nMaxTrackbarNum;
int g_nContrastValue=80; //对比度值  
int g_nBrightValue=80;  //亮度值 
cv::Mat g_srcImage; 
cv::Mat g_templateImage; 
cv::Mat g_resultImage;
void OpenCV_Function::warpAffine()  {  
	 //【1】参数准备  
    //定义两组点，代表两个三角形  
    cv::Point2f srcTriangle[3];  
    cv::Point2f dstTriangle[3];  
    //定义一些Mat变量  
    cv::Mat rotMat( 2, 3, CV_32FC1 );  
    cv::Mat warpMat( 2, 3, CV_32FC1 );  

    //【2】加载源图像并作一些初始化  
    g_srcImage=cv::imread("girl-t1.jpg");
    // 设置目标图像的大小和类型与源图像一致  
    g_resultImage = cv::Mat::zeros( g_srcImage.rows, g_srcImage.cols, g_srcImage.type() );  
  
    //【3】设置源图像和目标图像上的三组点以计算仿射变换  
    srcTriangle[0] = cv::Point2f( 0,0 );  
    srcTriangle[1] = cv::Point2f( static_cast<float>(g_srcImage.cols - 1), 0 );  
    srcTriangle[2] = cv::Point2f( 0, static_cast<float>(g_srcImage.rows - 1 ));  
  
    dstTriangle[0] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.0), static_cast<float>(g_srcImage.rows*0.33));  
    dstTriangle[1] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.65), static_cast<float>(g_srcImage.rows*0.35));  
    dstTriangle[2] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.15), static_cast<float>(g_srcImage.rows*0.6));  
  
    //【4】求得仿射变换  
    warpMat = getAffineTransform( srcTriangle, dstTriangle );  
    //【5】对源图像应用刚刚求得的仿射变换  
   cv::warpAffine( g_srcImage, g_resultImage, warpMat, g_resultImage.size() );  
  
    //【6】对图像进行缩放后再旋转  
    cv::Point center = cv::Point( g_srcImage.cols/2, g_srcImage.rows/2 );  
    double angle = -20.0;  
    double scale = 1;  
    // 通过上面的旋转细节信息求得旋转矩阵  
    rotMat =cv::getRotationMatrix2D( center, angle, scale );  
    // 旋转已缩放后的图像  
    cv::warpAffine( g_resultImage, g_templateImage, rotMat, g_resultImage.size() );  
	cv::imshow("【原始图】",g_srcImage);  
	cv::imshow( "g_templateImage", g_templateImage );
    cv::imshow( "g_resultImage", g_resultImage );  
}  
void OpenCV_Function::rotateImage()  {  
    g_srcImage=cv::imread("girl-t1.jpg");
  
    cv::Point center = cv::Point( g_srcImage.cols/2, g_srcImage.rows/2 );  
    double angle = -20.0;  
    double scale = 1;  
    cv::Mat rotMat =cv::getRotationMatrix2D( center, angle, scale );  
    cv::warpAffine( g_srcImage, g_resultImage, rotMat, g_srcImage.size() );  

	cv::imshow("【原始图】",g_srcImage);  
    cv::imshow( "g_resultImage", g_resultImage );  
}  

void OpenCV_Function::detector11(){
	g_srcImage=cv::imread("girl-t1.jpg");
	int minHessian = 40;//定义SURF中的hessian阈值特征点检测算子 

	cv::SurfFeatureDetector detector( minHessian );//定义一个SurfFeatureDetector（SURF） 特征检测类对象  
    std::vector<cv::KeyPoint> keypoints_1;//vector模板类是能够存放任意类型的动态数组，能够增加和压缩数据  
	 //【3】调用detect函数检测出SURF特征关键点，保存在vector容器中  
    detector.detect( g_srcImage, keypoints_1 );  

	//【4】绘制特征关键点  
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( g_srcImage, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );  
 
    //【5】显示效果图  
    cv::imshow("特征点检测效果图1", img_keypoints_1 );  

}
void OpenCV_Function::floodFill(){
	g_srcImage=cv::imread("girl-t1.jpg");
    cv::imshow("【原始图】",g_srcImage);  
    cv::Rect ccomp;  
    cv::floodFill(g_srcImage, cv::Point(50,300), cv::Scalar(155, 255,55), &ccomp, cv::Scalar(20, 20, 20),cv::Scalar(20, 20, 20));  
    cv::imshow("【效果图】",g_srcImage);  
}
//“如果某一点在任意方向的一个微小变动都会引起灰度很大的变化，那么我们就把它称之为角点”
//在当前的图像处理领域，角点检测算法可归纳为三类：
    //<1>基于灰度图像的角点检测
    //<2>基于二值图像的角点检测
    //<3>基于轮廓曲线的角点检测
void OpenCV_Function::cornerHarris(){
	g_srcImage=cv::imread("girl-t1.jpg");
	cv::cvtColor( g_srcImage, g_templateImage, CV_BGR2GRAY ); 
	g_resultImage = cv::Mat::zeros( g_srcImage.size(), CV_32FC1 );  
	cv::Mat normImage,scaledImage;//归一化后的图
	//进行角点检测  
	cv::cornerHarris(g_templateImage,g_resultImage, 2, 3, 0.04);

	cv::normalize(g_resultImage,normImage,0,255,cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs( normImage, scaledImage );//将归一化后的图线性变换成8位无符号整型   
	 // 将检测到的，且符合阈值条件的角点绘制出来  
    for( int j = 0; j < normImage.rows ; j++ )  { 
		for( int i = 0; i < normImage.cols; i++ )  {  
			if( (int) normImage.at<float>(j,i) > 180 )  {  
				cv::circle( g_templateImage, cv::Point( i, j ), 5,  cv::Scalar(10,10,255), 2, 8, 0 );  
				cv::circle( scaledImage, cv::Point( i, j ), 5, cv:: Scalar(0,10,255), 2, 8, 0 );  
			}  
		}  
    }  
    //---------------------------【4】显示最终效果---------------------------------  
	cv::imshow("g_templateImage",g_templateImage);
	cv::imshow("convertScaleAbs",scaledImage);

	cv::imshow(WINDOW_NAME1,g_srcImage);
}
/**
OpenCV中的霍夫线变换有如下三种：
	<1>标准霍夫变换（StandardHough Transform，SHT），由HoughLines函数调用。
	<2>多尺度霍夫变换（Multi-ScaleHough Transform，MSHT），由HoughLines函数调用。
	<3>累计概率霍夫变换（ProgressiveProbabilistic Hough Transform，PPHT），由HoughLinesP函数调用。

霍夫梯度法的原理是这样的。
	【1】首先对图像应用边缘检测，比如用canny边缘检测。
	【2】然后，对边缘图像中的每一个非零点，考虑其局部梯度，即用Sobel（）函数计算x和y方向的Sobel一阶导数得到梯度。
	【3】利用得到的梯度，由斜率指定的直线上的每一个点都在累加器中被累加，这里的斜率是从一个指定的最小值到指定的最大值的距离。
	【4】同时，标记边缘图像中每一个非0像素的位置。
	【5】然后从二维累加器中这些点中选择候选的中心，这些中心都大于给定阈值并且大于其所有近邻。这些候选的中心按照累加值降序排列，以便于最支持像素的中心首先出现。
	【6】接下来对每一个中心，考虑所有的非0像素。
	【7】这些像素按照其与中心的距离排序。从到最大半径的最小距离算起，选择非0像素最支持的一条半径。
	【8】如果一个中心收到边缘图像非0像素最充分的支持，并且到前期被选择的中心有足够的距离，那么它就会被保留下来。

霍夫梯度法的缺点
	<1>在霍夫梯度法中，我们使用Sobel导数来计算局部梯度，那么随之而来的假设是，其可以视作等同于一条局部切线，并这个不是一个数值稳定的做法。
		在大多数情况下，这样做会得到正确的结果，但或许会在输出中产生一些噪声。
	<2>在边缘图像中的整个非0像素集被看做每个中心的候选部分。因此，如果把累加器的阈值设置偏低，算法将要消耗比较长的时间。
		第三，因为每一个中心只选择一个圆，如果有同心圆，就只能选择其中的一个。
	<3>因为中心是按照其关联的累加器值的升序排列的，并且如果新的中心过于接近之前已经接受的中心的话，就不会被保留下来。
		且当有许多同心圆或者是近似的同心圆时，霍夫梯度法的倾向是保留最大的一个圆。可以说这是一种比较极端的做法，因为在这里默认Sobel导数会产生噪声，若是对于无穷分辨率的平滑图像而言的话，这才是必须的。
*/
void HoughCircles(){
	cv::Mat srcImage = cv::imread("20111019_e9ded922ab2b02875d1fvv3lYiYIYExI.jpg"); 
    cv::Mat midImage,dstImage;//临时变量和目标图的定义 

	   //【3】转为灰度图，进行图像平滑  
    cv::cvtColor(srcImage,midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图  
    cv::GaussianBlur( midImage, midImage, cv::Size(9, 9), 2, 2 );  

	 //【4】进行霍夫圆变换  
    cv::vector<cv::Vec3f> circles;  
    cv::HoughCircles( midImage, circles, CV_HOUGH_GRADIENT,1.5, 10, 100, 100, 0, 0 );  

	 //【5】依次在图中绘制出圆  
    for( size_t i = 0; i < circles.size(); i++ )  
    {  
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));  
        int radius = cvRound(circles[i][2]);  
        //绘制圆心  
        cv::circle( srcImage, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );  
        //绘制圆轮廓  
        cv::circle( srcImage, center, radius, cv::Scalar(155,50,255), 1, 8, 0 );  
    }  
	 //【6】显示效果图    
    cv::imshow("【HoughCircles效果图】", srcImage);    
}
void HoughLinesP(){
	cv::Mat srcImage = cv::imread("girl-t1.jpg");    
    cv::Mat midImage,dstImage;//临时变量和目标图的定义 

	//【2】进行边缘检测和转化为灰度图  
    cv::Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测  
    cv::cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图  
	//【3】进行霍夫线变换  
    cv::vector<cv::Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
    cv::HoughLinesP(midImage, lines, 1, CV_PI/180, 150, 0, 0 ); 

	 //【4】依次在图中绘制出每条线段  
    for( size_t i = 0; i < lines.size(); i++ )  
    {  
        cv::Vec4i l = lines[i];  
        line( dstImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(86,88,155), 10, CV_AA);  
    }   
    //【6】边缘检测后的图   
    cv::imshow("【HoughLinesP边缘检测后的图】", midImage);    
    cv::imshow("【HoughLinesP效果图】", dstImage);  
}
void OpenCV_Function::Hough(){
	cv::Mat srcImage = cv::imread("girl-t1.jpg");  
    cv::Mat midImage,dstImage;//临时变量和目标图的定义 

	//【2】进行边缘检测和转化为灰度图  
    cv::Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测  
    cv::cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图  
	//【3】进行霍夫线变换  
    cv::vector<cv::Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
    cv::HoughLines(midImage, lines, 1, CV_PI/180, 150, 0, 0 ); 

	for( size_t i = 0; i < lines.size(); i++ )  
    {  
        float rho = lines[i][0], theta = lines[i][1];  
        cv::Point pt1, pt2;  
        double a = cos(theta), b = sin(theta);  
        double x0 = a*rho, y0 = b*rho;  
        pt1.x = cvRound(x0 + 1000*(-b));  
        pt1.y = cvRound(y0 + 1000*(a));  
        pt2.x = cvRound(x0 - 1000*(-b));  
        pt2.y = cvRound(y0 - 1000*(a));  
        line( dstImage, pt1, pt2, cv::Scalar(55,100,195), 2, CV_AA);  
    }  
    //【5】显示原始图    
    cv::imshow("【原始图】", srcImage);    
    //【6】边缘检测后的图   
    cv::imshow("【HoughLines边缘检测后的图】", midImage);    
    cv::imshow("【HoughLines效果图】", dstImage);  
	

	 HoughLinesP();

	 HoughCircles();
}

void OpenCV_Function::resize(){
	g_srcImage=cv::imread("girl-t1.jpg");

	cv::pyrDown(g_srcImage,g_resultImage);  
	cv::imshow( "pyrDown", g_resultImage );  

	cv::pyrUp(g_srcImage,g_resultImage);  
	cv::imshow( "pyrUp", g_resultImage );  

	cv::resize(g_srcImage,g_resultImage,cv::Size( g_srcImage.cols*2, g_srcImage.rows*2 ));  
	cv::imshow( "resize*2", g_resultImage );  

	cv::resize(g_srcImage,g_resultImage,cv::Size( g_srcImage.cols/2, g_srcImage.rows/2 ));  
	cv::imshow( "resize/2", g_resultImage );   

	cv::imshow( WINDOW_NAME1, g_srcImage );
}

void OpenCV_Function::filter(){
	g_srcImage=cv::imread("girl-t1.jpg");

	//均值滤波（邻域平均滤波）
	cv::blur(g_srcImage,g_resultImage,cv::Size(2,2)); //值越大越模糊
	cv::imshow( "blur", g_resultImage );

	//高斯滤波
	cv::GaussianBlur( g_srcImage, g_resultImage, cv::Size( 99, 99 ), 0, 0 );   //值越大越模糊,且值只能是正数和奇数
	cv::imshow("GaussianBlur", g_resultImage );

	//方框滤波
	cv::boxFilter(g_srcImage,g_resultImage,-1,cv::Size(5,5));//
	cv::imshow("boxFilter", g_resultImage );

	//中值滤波
	cv::medianBlur(g_srcImage,g_resultImage,9);//这个参数必须是大于1的奇数
	cv::imshow("medianBlur", g_resultImage );

	//bilateralFilter 双边滤波器
	cv::bilateralFilter( g_srcImage, g_resultImage, 25, 25*2, 25/2 );  
	cv::imshow("bilateralFilter", g_resultImage );

	//erode函数，使用像素邻域内的局部极小运算符来腐蚀一张图片
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));  
	cv::erode(g_srcImage,g_resultImage,element,cv::Point(-1,-1),1);
	cv::imshow("erode", g_resultImage );
	//dilate函数，使用像素邻域内的局部极大运算符来膨胀一张图片
	cv::dilate(g_srcImage,g_resultImage,element);
	cv::imshow("dilate", g_resultImage );

	//开运算（Opening Operation），其实就是先腐蚀后膨胀的过程
	//dst=open(src,element)=dilate(erode(src,element));

	//闭运算(Closing Operation),  其实就是先膨胀后腐蚀的过程
	//dst=close(src,element)=erode(dilate(src,element));

	//形态学梯度（Morphological Gradient）为膨胀图与腐蚀图之差
	//dst=morph_grad(src,element)=dilate(src,element)-erode(src,element);

	//顶帽运算（Top Hat）又常常被译为”礼帽“运算。为原图像与上文刚刚介绍的“开运算“的结果图之差
	//dst=src-open(src,element);

	//黑帽（Black Hat）运算为”闭运算“的结果图与原图像之差。
	//dst=close(src,element)-src;
	cv::morphologyEx(g_srcImage,g_resultImage, cv::MORPH_OPEN, element);  
	cv::imshow( "morphologyEx", g_resultImage );

	//最简单的canny用法，拿到原图后直接用。  
	//这个函数阈值1和阈值2两者的小者用于边缘连接，而大者用来控制强边缘的初始段，推荐的高低阈值比在2:1到3:1之间。
	cv::Mat cannyMat=g_srcImage.clone();
	cv::Canny(cannyMat,cannyMat,3,9);
	cv::imshow( "Canny", cannyMat );
	//----------------------------------------------------------------------------------  
    //  二、高阶的canny用法，转成灰度图，降噪，用canny，最后将得到的边缘作为掩码，拷贝原图到效果图上，得到彩色的边缘图  
    //----------------------------------------------------------------------------------  
	cv::Mat dst,edge,gray; 
	dst.create( g_srcImage.size(), g_srcImage.type() );   // 【1】创建与src同类型和大小的矩阵(dst)  
	cv::cvtColor(g_srcImage,gray,CV_BGR2GRAY);// 【2】将原图像转换为灰度图像  
    cv::blur( gray, edge, cv::Size(3,3) );  // 【3】先用使用 3x3内核来降噪  
	cv::Canny(edge,edge,3,9);
	dst = cv::Scalar::all(0);   //【5】将dst内的所有元素设置为0   
	g_srcImage.copyTo( dst, edge); //【6】使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中  
	cv::imshow( "Canny2", dst );
	
	//----------------------------------------------------------------------------------  
    //  调用Sobel函数的实例代码
    //----------------------------------------------------------------------------------  
	cv::Mat grad_x, grad_y;  
   cv::Mat abs_grad_x, abs_grad_y;  
	 //【3】求 X方向梯度  
    cv::Sobel( g_srcImage, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_x, abs_grad_x );  
    cv::imshow("【效果图】 X方向Sobel", abs_grad_x);   
  
    //【4】求Y方向梯度  
    cv::Sobel( g_srcImage, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_y, abs_grad_y );  
    cv::imshow("【效果图】Y方向Sobel", abs_grad_y);   
  
    //【5】合并梯度(近似)  
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );  
    cv::imshow("【效果图】整体方向Sobel", dst);   
  
	//----------------------------------------------------------------------------------  
    // Laplacian 算子是n维欧几里德空间中的一个二阶微分算子，定义为梯度grad（）的散度div（）。因此如果f是二阶可微的实函数，则f的拉普拉斯算子定义为：
    //----------------------------------------------------------------------------------  
	cv::Laplacian( g_srcImage, dst, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT );  
	cv::convertScaleAbs( dst, abs_grad_y );  
	cv::imshow("Laplacian", abs_grad_y); 
	

	//----------------------------------------------------------------------------------  
    //  scharr一般我就直接称它为滤波器，而不是算子。上文我们已经讲到，它在OpenCV中主要是配合Sobel算子的运算而存在的,
    //----------------------------------------------------------------------------------  
	 //【3】求 X方向梯度  
	cv::Scharr( g_srcImage, grad_x, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT );  
    cv:: convertScaleAbs( grad_x, abs_grad_x );  
    cv::imshow("【效果图】 X方向Scharr", abs_grad_x);   
	//【4】求Y方向梯度  
	cv::Scharr( g_srcImage, grad_y, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_y, abs_grad_y );  
    cv::imshow("【效果图】Y方向Scharr", abs_grad_y);

	cv::imshow( WINDOW_NAME1, g_srcImage );
}

//改变图像对比度和亮度值的回调函数  
void contrastAndBright(int,void*){
	for(int y=0;y<g_srcImage.rows;y++){
		for(int x=0;x<g_srcImage.cols;x++){
			for(int c = 0; c < 3; c++ )  {
				g_resultImage.at<cv::Vec3b>(y,x)[c]= cv::saturate_cast<uchar>( (g_nContrastValue*0.01)*(g_srcImage.at<cv::Vec3b>(y,x)[c] ) + g_nBrightValue );  
			}
		}
	}
	 //显示图像  
    cv::imshow(WINDOW_NAME1, g_srcImage);  
    cv::imshow(WINDOW_NAME2, g_resultImage); 
}
//改变图像对比度和亮度值的主方法  
void OpenCV_Function::contrastAndBrightByTrackbar(){
	system("color5F");   
	g_srcImage=cv::imread("girl-t1.jpg");

	g_resultImage= cv::Mat::zeros( g_srcImage.size(), g_srcImage.type());

	cv::namedWindow( WINDOW_NAME1, cv::WINDOW_AUTOSIZE );
	cv::namedWindow( WINDOW_NAME2, cv::WINDOW_AUTOSIZE );

	cv::createTrackbar("对比度：", WINDOW_NAME2,&g_nContrastValue,300,contrastAndBright );  
    cv::createTrackbar("亮   度：",WINDOW_NAME2,&g_nBrightValue,200,contrastAndBright );  

	contrastAndBright(g_nContrastValue,0);  
    contrastAndBright(g_nBrightValue,0);  


}
//这个内部方法要放在调用方法前面，否则编译不通过。
void on_Matching(int,void* ){
	//【1】给局部变量初始化
	cv::Mat srcImage;
	g_srcImage.copyTo( srcImage );

	//【2】初始化用于结果输出的矩阵
	int resultImage_cols =  g_srcImage.cols - g_templateImage.cols + 1;
	int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
	g_resultImage.create( resultImage_cols, resultImage_rows, CV_32FC1 );

	//【3】进行匹配和标准化
	cv::matchTemplate( g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod );
	cv::normalize( g_resultImage, g_resultImage, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	//【4】通过函数 minMaxLoc 定位最匹配的位置
	double minValue; double maxValue; cv::Point minLocation; cv::Point maxLocation;
	cv::Point matchLocation;
	cv::minMaxLoc( g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, cv::Mat() );

	//【5】对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值有着更高的匹配结果. 而其余的方法, 数值越大匹配效果越好
	//此句代码的OpenCV2版为：
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//此句代码的OpenCV3版为：
	if( g_nMatchMethod  == cv::TM_SQDIFF || g_nMatchMethod == cv::TM_SQDIFF_NORMED )
	{ matchLocation = minLocation; }
	else
	{ matchLocation = maxLocation; }

	//【6】绘制出矩形，并显示最终结果
	cv::rectangle( srcImage, matchLocation, cv::Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ),cv::Scalar(0,0,255), 2, 8, 0 );
	cv::rectangle( g_resultImage, matchLocation,cv::Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), cv::Scalar(0,0,255), 2, 8, 0 );

	cv::imshow( WINDOW_NAME1, srcImage );
	cv::imshow( WINDOW_NAME2, g_resultImage );
}

//模板匹配示例,通过滑动条控制方法
void OpenCV_Function::matchByTrackbar(){
	//【0】改变console字体颜色
	system("color 1F"); 

	g_nMaxTrackbarNum = 5;

	g_srcImage = cv::imread( "girl-t1.jpg", 1 );
	g_templateImage = cv::imread( "girl-logo-t1.jpg", 1 );

	cv::namedWindow( WINDOW_NAME1, cv::WINDOW_AUTOSIZE );
	cv::namedWindow( WINDOW_NAME2, cv::WINDOW_AUTOSIZE );

	cv::createTrackbar( "方法", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum,on_Matching);
	on_Matching(0,0);
}

//分割通道，然后再混合。
void OpenCV_Function::splitiChannelBlending(){
	cv::Mat logoImg=cv::imread("dota_logo.jpg",0);
	cv::vector<cv::Mat> channels; 
	
	for(int i=0;i<3;i++){
		cv::Mat srcImg=cv::imread("dota.jpg");

		cv::split(srcImg,channels);

		cv::Mat imageChannel=channels.at(i);  

		cv::Mat newImg=imageChannel(cv::Rect(800,350,logoImg.cols,logoImg.rows));

		cv::addWeighted(newImg,1,logoImg,0.5,0.,newImg);  

		cv::merge(channels,srcImg);  

		cv::namedWindow("newImg"+i);
		cv::imshow("newImg"+i,srcImg);
	}
}
//线性混合
void OpenCV_Function:: LinearBlending(){
	double alphaValue=0.5;
	double betaValue=1.0-alphaValue;

	cv::Mat mogu,rain,dstImg;

	mogu=cv::imread("mogu.jpg");
	rain=cv::imread("rain.jpg");
	if(!mogu.data ) { printf("你妹，读取img1错误~！ \n"); return ; }
	if(!rain.data ) { printf("你妹，读取img2错误~！ \n"); return ; }

	cv::addWeighted(mogu,alphaValue,rain,betaValue,0,dstImg);

	cv::namedWindow("rain");
	cv::imshow("rain",rain);

	cv::namedWindow("mogu");
	cv::imshow("mogu",mogu);

	cv::namedWindow("LinearBlending");
	cv::imshow("LinearBlending",dstImg);

}
//测试加载图像，高宽缩放，图像混合，灰度蒙层
void OpenCV_Function:: Load_Show(){
	cv::Mat bg=cv::imread("dota.jpg");
	cv::Mat logo=cv::imread("dota_logo.jpg");
	cv::Mat girl=cv::imread("girl.jpg");

	double scale=0.3;
	cv::Size dsize = cv::Size(girl.cols*scale,girl.rows*scale);
	cv::Mat newGirl = cv::Mat(dsize,CV_32S);
	cv::resize(girl, newGirl,dsize);
	cv::namedWindow("newGirl");
	cv::imshow("newGirl",newGirl);

	cv::Mat newImg2=bg(cv::Rect(50,350,newGirl.cols,newGirl.rows));
	cv::addWeighted(newImg2,0.1,newGirl,0.8,0,newImg2);  

	cv::Mat newImg=bg(cv::Rect(800,350,logo.cols,logo.rows));
	cv::addWeighted(newImg,0.1,logo,0.5,0,newImg);  

	cv::namedWindow("newImg");
	cv::imshow("newImg",bg);

}
//----------------------------------【ROI_AddImage( )函数】----------------------------------
// 函数名：ROI_AddImage（）
//     描述：利用感兴趣区域ROI实现图像叠加  //通过copyto方式，显示内嵌的图片。s
//----------------------------------------------------------------------------------------------
void OpenCV_Function:: ROI_AddImage(){
	//【1】读入图像
	cv::Mat srcImage1= cv::imread("dota.jpg");
	cv::Mat logoImage= cv::imread("dota_logo.jpg");
	if(!srcImage1.data ) { printf("你妹，读取srcImage1错误~！ \n"); return ; }
	if(!logoImage.data ) { printf("你妹，读取logoImage错误~！ \n"); return ; }
	//【2】定义一个Mat类型并给其设定ROI区域
	cv::Mat imageROI= srcImage1(cv::Rect(800,350,logoImage.cols,logoImage.rows));
	//【3】加载掩模（必须是灰度图）
	cv::Mat mask= cv::imread("dota_logo.jpg",1);
	//【4】将掩膜拷贝到ROI
	logoImage.copyTo(imageROI,mask);
	//【5】显示结果
	cv::namedWindow("<1>利用ROI实现图像叠加示例窗口");
	cv::imshow("<1>利用ROI实现图像叠加示例窗口",srcImage1);

}