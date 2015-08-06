#pragma once

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "iostream"
using namespace cv;

// 全局二值化实现类 
class Globle_Otsu_Binarizer 
{
public:

	double  Thoushold_Value;

	float   background_value;

	float   Percent10_upper_by_Thoushold_Value_index;

public :

	Globle_Otsu_Binarizer();

	virtual void  Binary_way(const Mat& input ,Mat & out_put) ;

};