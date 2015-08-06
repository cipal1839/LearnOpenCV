#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "Globle_Otsu_Binarizer.h"
#include "OpenCV_Function.h"

#define WINDOW_NAME1 "【原始图片】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【匹配窗口】"        //为窗口标题定义的宏 

int g_nMatchMethod;
int g_nMaxTrackbarNum;
cv::Mat g_srcImage; 
cv::Mat g_templateImage; 
cv::Mat g_resultImage;


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
	cv::normalize( g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat() );

	//【4】通过函数 minMaxLoc 定位最匹配的位置
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	cv::minMaxLoc( g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat() );

	//【5】对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值有着更高的匹配结果. 而其余的方法, 数值越大匹配效果越好
	//此句代码的OpenCV2版为：
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//此句代码的OpenCV3版为：
	if( g_nMatchMethod  == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED )
	{ matchLocation = minLocation; }
	else
	{ matchLocation = maxLocation; }

	//【6】绘制出矩形，并显示最终结果
	cv::rectangle( srcImage, matchLocation, Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), Scalar(0,0,255), 2, 8, 0 );
	cv::rectangle( g_resultImage, matchLocation, Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), Scalar(0,0,255), 2, 8, 0 );

	cv::imshow( WINDOW_NAME1, srcImage );
	cv::imshow( WINDOW_NAME2, g_resultImage );
}


//模板匹配示例,通过滑动条控制方法
void OpenCV_Function::matchByTrackbar(){
	//【0】改变console字体颜色
	system("color 1F"); 

	g_nMaxTrackbarNum = 5;

	g_srcImage = cv::imread( "exam001.jpg", 1 );
	g_templateImage = cv::imread( "exam-logo1.jpg", 1 );

	namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
	namedWindow( WINDOW_NAME2, WINDOW_AUTOSIZE );

	createTrackbar( "方法", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum,on_Matching);
	on_Matching(0,0);
}

//初次测试代码，调用了otsu 抓去二值化。
void OpenCV_Function::first()  
{  
	for(int i=0;i<5;i++){
		//string imgPath = "pic"+itoa(i)+".jpg";
		std::stringstream ss;
		ss << "pic" << i<<".jpg";
		// 读入一张图片（游戏原画）  
		cv::Mat inputImg=cv::imread(ss.str());  

		float Thousold_Gray;
		float Thousold_Blue; 
		float Thousold_Green; 
		float Thousold_Red;
		float Blue_Max_Background_value;
		float Green_Max_Background_value;
		float Red_Max_Background_value;
		float Blue_percent10_upper;
		float Green_percent10_upper;
		float Red_percent10_upper;

		cv::Mat Image_BGR[3];
		cv::Mat  Image_BGR_Binary_Virtual[3];

		cv::split(inputImg,Image_BGR);

		Globle_Otsu_Binarizer* point=new Globle_Otsu_Binarizer();
		// blue channel
		point->Binary_way(Image_BGR[0],Image_BGR_Binary_Virtual[0]);
		Thousold_Blue=point->Thoushold_Value;
		Blue_Max_Background_value=point->background_value;
		Blue_percent10_upper=point->Percent10_upper_by_Thoushold_Value_index;
		// green channel
		point->Binary_way(Image_BGR[1],Image_BGR_Binary_Virtual[1]);
		Thousold_Green=point->Thoushold_Value;
		Green_Max_Background_value=point->background_value;
		Green_percent10_upper=point->Percent10_upper_by_Thoushold_Value_index;
		// red channel 
		point->Binary_way(Image_BGR[2],Image_BGR_Binary_Virtual[2]);
		Thousold_Red=point->Thoushold_Value;
		Red_Max_Background_value=point->background_value;
		Red_percent10_upper=point->Percent10_upper_by_Thoushold_Value_index;

		delete point;

		std::cout <<"Thousold_Blue:"<<Thousold_Blue <<" Blue_Max_Background_value:"<<Blue_Max_Background_value << std::endl;
		std::cout <<"Thousold_Green:"<<Thousold_Green <<" Green_Max_Background_value:"<<Green_Max_Background_value << std::endl;
		std::cout <<"Thousold_Red:"<<Thousold_Red <<" Red_Max_Background_value:"<<Red_Max_Background_value << std::endl;

		cv::namedWindow("游戏原画-0");   
		cv::imshow("游戏原画-0",Image_BGR[0]); 

		cv::imshow("游戏原画-1",Image_BGR[1]);  
		cv::imshow("游戏原画-1",Image_BGR[2]);  

	}
}

//分割通道，然后再混合。
void OpenCV_Function::splitiChannelBlending(){
	cv::Mat logoImg=cv::imread("dota_logo.jpg",0);
	for(int i=0;i<3;i++){
		cv::Mat srcImg=cv::imread("dota.jpg");
		cv::vector<cv::Mat> channels; 
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