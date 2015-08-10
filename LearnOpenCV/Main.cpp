#include <opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  

#include "OpenCV_Function.h"

int main(){
	//ROI_AddImage();
    //Load_Show();
	//LinearBlending();
	OpenCV_Function* of=new OpenCV_Function();

	of->contrastAndBrightByTrackbar();

	cv::waitKey(); 
	return 0; 
}