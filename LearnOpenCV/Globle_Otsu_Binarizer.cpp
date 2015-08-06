#include "Globle_Otsu_Binarizer.h"

Globle_Otsu_Binarizer::Globle_Otsu_Binarizer():Thoushold_Value(0),background_value(0),Percent10_upper_by_Thoushold_Value_index(0)
{

}
//  大津方法的二值化 ,依赖opencv 的方法 
void  Globle_Otsu_Binarizer::Binary_way(const Mat & _src ,Mat& out_put) 
{   
	// The code is copy from the Opencv'S otsu Binary algorithm
    Size size = _src.size();
    if( _src.isContinuous() )// Fast the point search speed 
    {
        size.width *= size.height;
        size.height = 1;
    }
    const int N = 256;
    int i, j, h[N] = {0};

	// calculate the histograme ot he input image 
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.data + _src.step*i;
        j = 0;
   
        for( ; j <= size.width - 4; j += 4 )
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
      
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }
	// Find the max backgrond value ,which as a larger peak present 

	float pixs_number=h[0];
	for(int index=1;index<256;index++)
	{
	     if (pixs_number<h[index])
		 {
		   pixs_number=h[index];
		   background_value=index;
		 }
	}

	// calculate the biggest varience between the two clster 
    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0.0, q1 = 0.0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1.0 - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1.0 - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }
    Thoushold_Value= max_val;
	// calculate the Percent10_upper_by_Thoushold_Value_index value
	//float Percent_Thoushold_VS_MaxHeight=(float)h[int(Thoushold_Value)]/(float)h[int(background_value)];
	float distance =background_value-Thoushold_Value;
	Percent10_upper_by_Thoushold_Value_index=Thoushold_Value+0.1*distance;
}

