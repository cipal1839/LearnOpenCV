#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "OpenCV_Function.h"

#define WINDOW_NAME1 "��ԭʼͼƬ��"        //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "��ƥ�䴰�ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 

int g_nMatchMethod;
int g_nMaxTrackbarNum;
cv::Mat g_srcImage; 
cv::Mat g_templateImage; 
cv::Mat g_resultImage;


//����ڲ�����Ҫ���ڵ��÷���ǰ�棬������벻ͨ����
void on_Matching(int,void* ){
	//��1�����ֲ�������ʼ��
	cv::Mat srcImage;
	g_srcImage.copyTo( srcImage );

	//��2����ʼ�����ڽ������ľ���
	int resultImage_cols =  g_srcImage.cols - g_templateImage.cols + 1;
	int resultImage_rows = g_srcImage.rows - g_templateImage.rows + 1;
	g_resultImage.create( resultImage_cols, resultImage_rows, CV_32FC1 );

	//��3������ƥ��ͱ�׼��
	cv::matchTemplate( g_srcImage, g_templateImage, g_resultImage, g_nMatchMethod );
	cv::normalize( g_resultImage, g_resultImage, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	//��4��ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minValue; double maxValue; cv::Point minLocation; cv::Point maxLocation;
	cv::Point matchLocation;
	cv::minMaxLoc( g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, cv::Mat() );

	//��5�����ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ���Ÿ��ߵ�ƥ����. ������ķ���, ��ֵԽ��ƥ��Ч��Խ��
	//�˾�����OpenCV2��Ϊ��
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//�˾�����OpenCV3��Ϊ��
	if( g_nMatchMethod  == cv::TM_SQDIFF || g_nMatchMethod == cv::TM_SQDIFF_NORMED )
	{ matchLocation = minLocation; }
	else
	{ matchLocation = maxLocation; }

	//��6�����Ƴ����Σ�����ʾ���ս��
	cv::rectangle( srcImage, matchLocation, cv::Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ),cv::Scalar(0,0,255), 2, 8, 0 );
	cv::rectangle( g_resultImage, matchLocation,cv::Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), cv::Scalar(0,0,255), 2, 8, 0 );

	cv::imshow( WINDOW_NAME1, srcImage );
	cv::imshow( WINDOW_NAME2, g_resultImage );
}


//ģ��ƥ��ʾ��,ͨ�����������Ʒ���
void OpenCV_Function::matchByTrackbar(){
	//��0���ı�console������ɫ
	system("color 1F"); 

	g_nMaxTrackbarNum = 5;

	g_srcImage = cv::imread( "exam001.jpg", 1 );
	g_templateImage = cv::imread( "exam-logo1.jpg", 1 );

	cv::namedWindow( WINDOW_NAME1, cv::WINDOW_AUTOSIZE );
	cv::namedWindow( WINDOW_NAME2, cv::WINDOW_AUTOSIZE );

	cv::createTrackbar( "����", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum,on_Matching);
	on_Matching(0,0);
}

//�ָ�ͨ����Ȼ���ٻ�ϡ�
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
//���Ի��
void OpenCV_Function:: LinearBlending(){
	double alphaValue=0.5;
	double betaValue=1.0-alphaValue;

	cv::Mat mogu,rain,dstImg;

	mogu=cv::imread("mogu.jpg");
	rain=cv::imread("rain.jpg");
	if(!mogu.data ) { printf("���ã���ȡimg1����~�� \n"); return ; }
	if(!rain.data ) { printf("���ã���ȡimg2����~�� \n"); return ; }

	cv::addWeighted(mogu,alphaValue,rain,betaValue,0,dstImg);

	cv::namedWindow("rain");
	cv::imshow("rain",rain);

	cv::namedWindow("mogu");
	cv::imshow("mogu",mogu);

	cv::namedWindow("LinearBlending");
	cv::imshow("LinearBlending",dstImg);

}
//���Լ���ͼ�񣬸߿����ţ�ͼ���ϣ��Ҷ��ɲ�
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
//----------------------------------��ROI_AddImage( )������----------------------------------
// ��������ROI_AddImage����
//     ���������ø���Ȥ����ROIʵ��ͼ�����  //ͨ��copyto��ʽ����ʾ��Ƕ��ͼƬ��s
//----------------------------------------------------------------------------------------------
void OpenCV_Function:: ROI_AddImage(){
	//��1������ͼ��
	cv::Mat srcImage1= cv::imread("dota.jpg");
	cv::Mat logoImage= cv::imread("dota_logo.jpg");
	if(!srcImage1.data ) { printf("���ã���ȡsrcImage1����~�� \n"); return ; }
	if(!logoImage.data ) { printf("���ã���ȡlogoImage����~�� \n"); return ; }
	//��2������һ��Mat���Ͳ������趨ROI����
	cv::Mat imageROI= srcImage1(cv::Rect(800,350,logoImage.cols,logoImage.rows));
	//��3��������ģ�������ǻҶ�ͼ��
	cv::Mat mask= cv::imread("dota_logo.jpg",1);
	//��4������Ĥ������ROI
	logoImage.copyTo(imageROI,mask);
	//��5����ʾ���
	cv::namedWindow("<1>����ROIʵ��ͼ�����ʾ������");
	cv::imshow("<1>����ROIʵ��ͼ�����ʾ������",srcImage1);

}