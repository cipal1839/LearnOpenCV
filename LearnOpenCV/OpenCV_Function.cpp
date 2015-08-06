#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "Globle_Otsu_Binarizer.h"
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
	cv::normalize( g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat() );

	//��4��ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minValue; double maxValue; Point minLocation; Point maxLocation;
	Point matchLocation;
	cv::minMaxLoc( g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat() );

	//��5�����ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ���Ÿ��ߵ�ƥ����. ������ķ���, ��ֵԽ��ƥ��Ч��Խ��
	//�˾�����OpenCV2��Ϊ��
	//if( g_nMatchMethod  == CV_TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED )
	//�˾�����OpenCV3��Ϊ��
	if( g_nMatchMethod  == TM_SQDIFF || g_nMatchMethod == TM_SQDIFF_NORMED )
	{ matchLocation = minLocation; }
	else
	{ matchLocation = maxLocation; }

	//��6�����Ƴ����Σ�����ʾ���ս��
	cv::rectangle( srcImage, matchLocation, Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), Scalar(0,0,255), 2, 8, 0 );
	cv::rectangle( g_resultImage, matchLocation, Point( matchLocation.x + g_templateImage.cols , matchLocation.y + g_templateImage.rows ), Scalar(0,0,255), 2, 8, 0 );

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

	namedWindow( WINDOW_NAME1, WINDOW_AUTOSIZE );
	namedWindow( WINDOW_NAME2, WINDOW_AUTOSIZE );

	createTrackbar( "����", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum,on_Matching);
	on_Matching(0,0);
}

//���β��Դ��룬������otsu ץȥ��ֵ����
void OpenCV_Function::first()  
{  
	for(int i=0;i<5;i++){
		//string imgPath = "pic"+itoa(i)+".jpg";
		std::stringstream ss;
		ss << "pic" << i<<".jpg";
		// ����һ��ͼƬ����Ϸԭ����  
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

		cv::namedWindow("��Ϸԭ��-0");   
		cv::imshow("��Ϸԭ��-0",Image_BGR[0]); 

		cv::imshow("��Ϸԭ��-1",Image_BGR[1]);  
		cv::imshow("��Ϸԭ��-1",Image_BGR[2]);  

	}
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