#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv/highgui.h"
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


void OpenCV_Function::filter(){
	g_srcImage=cv::imread("girl-t1.jpg");
	
	cv::blur(g_srcImage,g_resultImage,cv::Size(2,2)); //值越大越模糊
	cv::imshow( "blur", g_resultImage );

	cv::GaussianBlur( g_srcImage, g_resultImage, cv::Size( 99, 99 ), 0, 0 );   //值越大越模糊,且值只能是正数和奇数
	cv::imshow("GaussianBlur", g_resultImage );

	cv::boxFilter(g_srcImage,g_resultImage,-1,cv::Size(5,5));
	cv::imshow("boxFilter", g_resultImage );

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