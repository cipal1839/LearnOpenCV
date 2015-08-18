#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include <iostream>  
#include "opencv/cv.h"
#include "OpenCV_Function.h"

#define WINDOW_NAME1 "��ԭʼͼƬ��"        //Ϊ���ڱ��ⶨ��ĺ� 
#define WINDOW_NAME2 "��Ч�����ڡ�"        //Ϊ���ڱ��ⶨ��ĺ� 

int g_nMatchMethod;
int g_nMaxTrackbarNum;
int g_nContrastValue=80; //�Աȶ�ֵ  
int g_nBrightValue=80;  //����ֵ 
cv::Mat g_srcImage; 
cv::Mat g_templateImage; 
cv::Mat g_resultImage;
void OpenCV_Function::warpAffine()  {  
	 //��1������׼��  
    //��������㣬��������������  
    cv::Point2f srcTriangle[3];  
    cv::Point2f dstTriangle[3];  
    //����һЩMat����  
    cv::Mat rotMat( 2, 3, CV_32FC1 );  
    cv::Mat warpMat( 2, 3, CV_32FC1 );  

    //��2������Դͼ����һЩ��ʼ��  
    g_srcImage=cv::imread("girl-t1.jpg");
    // ����Ŀ��ͼ��Ĵ�С��������Դͼ��һ��  
    g_resultImage = cv::Mat::zeros( g_srcImage.rows, g_srcImage.cols, g_srcImage.type() );  
  
    //��3������Դͼ���Ŀ��ͼ���ϵ�������Լ������任  
    srcTriangle[0] = cv::Point2f( 0,0 );  
    srcTriangle[1] = cv::Point2f( static_cast<float>(g_srcImage.cols - 1), 0 );  
    srcTriangle[2] = cv::Point2f( 0, static_cast<float>(g_srcImage.rows - 1 ));  
  
    dstTriangle[0] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.0), static_cast<float>(g_srcImage.rows*0.33));  
    dstTriangle[1] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.65), static_cast<float>(g_srcImage.rows*0.35));  
    dstTriangle[2] = cv::Point2f( static_cast<float>(g_srcImage.cols*0.15), static_cast<float>(g_srcImage.rows*0.6));  
  
    //��4����÷���任  
    warpMat = getAffineTransform( srcTriangle, dstTriangle );  
    //��5����Դͼ��Ӧ�øո���õķ���任  
   cv::warpAffine( g_srcImage, g_resultImage, warpMat, g_resultImage.size() );  
  
    //��6����ͼ��������ź�����ת  
    cv::Point center = cv::Point( g_srcImage.cols/2, g_srcImage.rows/2 );  
    double angle = -20.0;  
    double scale = 1;  
    // ͨ���������תϸ����Ϣ�����ת����  
    rotMat =cv::getRotationMatrix2D( center, angle, scale );  
    // ��ת�����ź��ͼ��  
    cv::warpAffine( g_resultImage, g_templateImage, rotMat, g_resultImage.size() );  
	cv::imshow("��ԭʼͼ��",g_srcImage);  
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

	cv::imshow("��ԭʼͼ��",g_srcImage);  
    cv::imshow( "g_resultImage", g_resultImage );  
}  

void OpenCV_Function::detector11(){
	g_srcImage=cv::imread("girl-t1.jpg");
	int minHessian = 40;//����SURF�е�hessian��ֵ������������ 

	cv::SurfFeatureDetector detector( minHessian );//����һ��SurfFeatureDetector��SURF�� ������������  
    std::vector<cv::KeyPoint> keypoints_1;//vectorģ�������ܹ�����������͵Ķ�̬���飬�ܹ����Ӻ�ѹ������  
	 //��3������detect��������SURF�����ؼ��㣬������vector������  
    detector.detect( g_srcImage, keypoints_1 );  

	//��4�����������ؼ���  
    cv::Mat img_keypoints_1;
    cv::drawKeypoints( g_srcImage, keypoints_1, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );  
 
    //��5����ʾЧ��ͼ  
    cv::imshow("��������Ч��ͼ1", img_keypoints_1 );  

}
void OpenCV_Function::floodFill(){
	g_srcImage=cv::imread("girl-t1.jpg");
    cv::imshow("��ԭʼͼ��",g_srcImage);  
    cv::Rect ccomp;  
    cv::floodFill(g_srcImage, cv::Point(50,300), cv::Scalar(155, 255,55), &ccomp, cv::Scalar(20, 20, 20),cv::Scalar(20, 20, 20));  
    cv::imshow("��Ч��ͼ��",g_srcImage);  
}
//�����ĳһ�������ⷽ���һ��΢С�䶯��������ҶȺܴ�ı仯����ô���ǾͰ�����֮Ϊ�ǵ㡱
//�ڵ�ǰ��ͼ�������򣬽ǵ����㷨�ɹ���Ϊ���ࣺ
    //<1>���ڻҶ�ͼ��Ľǵ���
    //<2>���ڶ�ֵͼ��Ľǵ���
    //<3>�����������ߵĽǵ���
void OpenCV_Function::cornerHarris(){
	g_srcImage=cv::imread("girl-t1.jpg");
	cv::cvtColor( g_srcImage, g_templateImage, CV_BGR2GRAY ); 
	g_resultImage = cv::Mat::zeros( g_srcImage.size(), CV_32FC1 );  
	cv::Mat normImage,scaledImage;//��һ�����ͼ
	//���нǵ���  
	cv::cornerHarris(g_templateImage,g_resultImage, 2, 3, 0.04);

	cv::normalize(g_resultImage,normImage,0,255,cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs( normImage, scaledImage );//����һ�����ͼ���Ա任��8λ�޷�������   
	 // ����⵽�ģ��ҷ�����ֵ�����Ľǵ���Ƴ���  
    for( int j = 0; j < normImage.rows ; j++ )  { 
		for( int i = 0; i < normImage.cols; i++ )  {  
			if( (int) normImage.at<float>(j,i) > 180 )  {  
				cv::circle( g_templateImage, cv::Point( i, j ), 5,  cv::Scalar(10,10,255), 2, 8, 0 );  
				cv::circle( scaledImage, cv::Point( i, j ), 5, cv:: Scalar(0,10,255), 2, 8, 0 );  
			}  
		}  
    }  
    //---------------------------��4����ʾ����Ч��---------------------------------  
	cv::imshow("g_templateImage",g_templateImage);
	cv::imshow("convertScaleAbs",scaledImage);

	cv::imshow(WINDOW_NAME1,g_srcImage);
}
/**
OpenCV�еĻ����߱任���������֣�
	<1>��׼����任��StandardHough Transform��SHT������HoughLines�������á�
	<2>��߶Ȼ���任��Multi-ScaleHough Transform��MSHT������HoughLines�������á�
	<3>�ۼƸ��ʻ���任��ProgressiveProbabilistic Hough Transform��PPHT������HoughLinesP�������á�

�����ݶȷ���ԭ���������ġ�
	��1�����ȶ�ͼ��Ӧ�ñ�Ե��⣬������canny��Ե��⡣
	��2��Ȼ�󣬶Ա�Եͼ���е�ÿһ������㣬������ֲ��ݶȣ�����Sobel������������x��y�����Sobelһ�׵����õ��ݶȡ�
	��3�����õõ����ݶȣ���б��ָ����ֱ���ϵ�ÿһ���㶼���ۼ����б��ۼӣ������б���Ǵ�һ��ָ������Сֵ��ָ�������ֵ�ľ��롣
	��4��ͬʱ����Ǳ�Եͼ����ÿһ����0���ص�λ�á�
	��5��Ȼ��Ӷ�ά�ۼ�������Щ����ѡ���ѡ�����ģ���Щ���Ķ����ڸ�����ֵ���Ҵ��������н��ڡ���Щ��ѡ�����İ����ۼ�ֵ�������У��Ա�����֧�����ص��������ȳ��֡�
	��6����������ÿһ�����ģ��������еķ�0���ء�
	��7����Щ���ذ����������ĵľ������򡣴ӵ����뾶����С��������ѡ���0������֧�ֵ�һ���뾶��
	��8�����һ�������յ���Եͼ���0�������ֵ�֧�֣����ҵ�ǰ�ڱ�ѡ����������㹻�ľ��룬��ô���ͻᱻ����������

�����ݶȷ���ȱ��
	<1>�ڻ����ݶȷ��У�����ʹ��Sobel����������ֲ��ݶȣ���ô��֮�����ļ����ǣ������������ͬ��һ���ֲ����ߣ����������һ����ֵ�ȶ���������
		�ڴ��������£���������õ���ȷ�Ľ�����������������в���һЩ������
	<2>�ڱ�Եͼ���е�������0���ؼ�������ÿ�����ĵĺ�ѡ���֡���ˣ�������ۼ�������ֵ����ƫ�ͣ��㷨��Ҫ���ıȽϳ���ʱ�䡣
		��������Ϊÿһ������ֻѡ��һ��Բ�������ͬ��Բ����ֻ��ѡ�����е�һ����
	<3>��Ϊ�����ǰ�����������ۼ���ֵ���������еģ���������µ����Ĺ��ڽӽ�֮ǰ�Ѿ����ܵ����ĵĻ����Ͳ��ᱻ����������
		�ҵ������ͬ��Բ�����ǽ��Ƶ�ͬ��Բʱ�������ݶȷ��������Ǳ�������һ��Բ������˵����һ�ֱȽϼ��˵���������Ϊ������Ĭ��Sobel������������������Ƕ�������ֱ��ʵ�ƽ��ͼ����ԵĻ�������Ǳ���ġ�
*/
void HoughCircles(){
	cv::Mat srcImage = cv::imread("20111019_e9ded922ab2b02875d1fvv3lYiYIYExI.jpg"); 
    cv::Mat midImage,dstImage;//��ʱ������Ŀ��ͼ�Ķ��� 

	   //��3��תΪ�Ҷ�ͼ������ͼ��ƽ��  
    cv::cvtColor(srcImage,midImage, CV_BGR2GRAY);//ת����Ե�����ͼΪ�Ҷ�ͼ  
    cv::GaussianBlur( midImage, midImage, cv::Size(9, 9), 2, 2 );  

	 //��4�����л���Բ�任  
    cv::vector<cv::Vec3f> circles;  
    cv::HoughCircles( midImage, circles, CV_HOUGH_GRADIENT,1.5, 10, 100, 100, 0, 0 );  

	 //��5��������ͼ�л��Ƴ�Բ  
    for( size_t i = 0; i < circles.size(); i++ )  
    {  
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));  
        int radius = cvRound(circles[i][2]);  
        //����Բ��  
        cv::circle( srcImage, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );  
        //����Բ����  
        cv::circle( srcImage, center, radius, cv::Scalar(155,50,255), 1, 8, 0 );  
    }  
	 //��6����ʾЧ��ͼ    
    cv::imshow("��HoughCirclesЧ��ͼ��", srcImage);    
}
void HoughLinesP(){
	cv::Mat srcImage = cv::imread("girl-t1.jpg");    
    cv::Mat midImage,dstImage;//��ʱ������Ŀ��ͼ�Ķ��� 

	//��2�����б�Ե����ת��Ϊ�Ҷ�ͼ  
    cv::Canny(srcImage, midImage, 50, 200, 3);//����һ��canny��Ե���  
    cv::cvtColor(midImage,dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ  
	//��3�����л����߱任  
    cv::vector<cv::Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������  
    cv::HoughLinesP(midImage, lines, 1, CV_PI/180, 150, 0, 0 ); 

	 //��4��������ͼ�л��Ƴ�ÿ���߶�  
    for( size_t i = 0; i < lines.size(); i++ )  
    {  
        cv::Vec4i l = lines[i];  
        line( dstImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(86,88,155), 10, CV_AA);  
    }   
    //��6����Ե�����ͼ   
    cv::imshow("��HoughLinesP��Ե�����ͼ��", midImage);    
    cv::imshow("��HoughLinesPЧ��ͼ��", dstImage);  
}
void OpenCV_Function::Hough(){
	cv::Mat srcImage = cv::imread("girl-t1.jpg");  
    cv::Mat midImage,dstImage;//��ʱ������Ŀ��ͼ�Ķ��� 

	//��2�����б�Ե����ת��Ϊ�Ҷ�ͼ  
    cv::Canny(srcImage, midImage, 50, 200, 3);//����һ��canny��Ե���  
    cv::cvtColor(midImage,dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ  
	//��3�����л����߱任  
    cv::vector<cv::Vec2f> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������  
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
    //��5����ʾԭʼͼ    
    cv::imshow("��ԭʼͼ��", srcImage);    
    //��6����Ե�����ͼ   
    cv::imshow("��HoughLines��Ե�����ͼ��", midImage);    
    cv::imshow("��HoughLinesЧ��ͼ��", dstImage);  
	

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

	//��ֵ�˲�������ƽ���˲���
	cv::blur(g_srcImage,g_resultImage,cv::Size(2,2)); //ֵԽ��Խģ��
	cv::imshow( "blur", g_resultImage );

	//��˹�˲�
	cv::GaussianBlur( g_srcImage, g_resultImage, cv::Size( 99, 99 ), 0, 0 );   //ֵԽ��Խģ��,��ֵֻ��������������
	cv::imshow("GaussianBlur", g_resultImage );

	//�����˲�
	cv::boxFilter(g_srcImage,g_resultImage,-1,cv::Size(5,5));//
	cv::imshow("boxFilter", g_resultImage );

	//��ֵ�˲�
	cv::medianBlur(g_srcImage,g_resultImage,9);//������������Ǵ���1������
	cv::imshow("medianBlur", g_resultImage );

	//bilateralFilter ˫���˲���
	cv::bilateralFilter( g_srcImage, g_resultImage, 25, 25*2, 25/2 );  
	cv::imshow("bilateralFilter", g_resultImage );

	//erode������ʹ�����������ڵľֲ���С���������ʴһ��ͼƬ
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));  
	cv::erode(g_srcImage,g_resultImage,element,cv::Point(-1,-1),1);
	cv::imshow("erode", g_resultImage );
	//dilate������ʹ�����������ڵľֲ����������������һ��ͼƬ
	cv::dilate(g_srcImage,g_resultImage,element);
	cv::imshow("dilate", g_resultImage );

	//�����㣨Opening Operation������ʵ�����ȸ�ʴ�����͵Ĺ���
	//dst=open(src,element)=dilate(erode(src,element));

	//������(Closing Operation),  ��ʵ���������ͺ�ʴ�Ĺ���
	//dst=close(src,element)=erode(dilate(src,element));

	//��̬ѧ�ݶȣ�Morphological Gradient��Ϊ����ͼ�븯ʴͼ֮��
	//dst=morph_grad(src,element)=dilate(src,element)-erode(src,element);

	//��ñ���㣨Top Hat���ֳ�������Ϊ����ñ�����㡣Ϊԭͼ�������ĸոս��ܵġ������㡰�Ľ��ͼ֮��
	//dst=src-open(src,element);

	//��ñ��Black Hat������Ϊ�������㡰�Ľ��ͼ��ԭͼ��֮�
	//dst=close(src,element)-src;
	cv::morphologyEx(g_srcImage,g_resultImage, cv::MORPH_OPEN, element);  
	cv::imshow( "morphologyEx", g_resultImage );

	//��򵥵�canny�÷����õ�ԭͼ��ֱ���á�  
	//���������ֵ1����ֵ2���ߵ�С�����ڱ�Ե���ӣ���������������ǿ��Ե�ĳ�ʼ�Σ��Ƽ��ĸߵ���ֵ����2:1��3:1֮�䡣
	cv::Mat cannyMat=g_srcImage.clone();
	cv::Canny(cannyMat,cannyMat,3,9);
	cv::imshow( "Canny", cannyMat );
	//----------------------------------------------------------------------------------  
    //  �����߽׵�canny�÷���ת�ɻҶ�ͼ�����룬��canny����󽫵õ��ı�Ե��Ϊ���룬����ԭͼ��Ч��ͼ�ϣ��õ���ɫ�ı�Եͼ  
    //----------------------------------------------------------------------------------  
	cv::Mat dst,edge,gray; 
	dst.create( g_srcImage.size(), g_srcImage.type() );   // ��1��������srcͬ���ͺʹ�С�ľ���(dst)  
	cv::cvtColor(g_srcImage,gray,CV_BGR2GRAY);// ��2����ԭͼ��ת��Ϊ�Ҷ�ͼ��  
    cv::blur( gray, edge, cv::Size(3,3) );  // ��3������ʹ�� 3x3�ں�������  
	cv::Canny(edge,edge,3,9);
	dst = cv::Scalar::all(0);   //��5����dst�ڵ�����Ԫ������Ϊ0   
	g_srcImage.copyTo( dst, edge); //��6��ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��  
	cv::imshow( "Canny2", dst );
	
	//----------------------------------------------------------------------------------  
    //  ����Sobel������ʵ������
    //----------------------------------------------------------------------------------  
	cv::Mat grad_x, grad_y;  
   cv::Mat abs_grad_x, abs_grad_y;  
	 //��3���� X�����ݶ�  
    cv::Sobel( g_srcImage, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_x, abs_grad_x );  
    cv::imshow("��Ч��ͼ�� X����Sobel", abs_grad_x);   
  
    //��4����Y�����ݶ�  
    cv::Sobel( g_srcImage, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_y, abs_grad_y );  
    cv::imshow("��Ч��ͼ��Y����Sobel", abs_grad_y);   
  
    //��5���ϲ��ݶ�(����)  
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );  
    cv::imshow("��Ч��ͼ�����巽��Sobel", dst);   
  
	//----------------------------------------------------------------------------------  
    // Laplacian ������nάŷ����¿ռ��е�һ������΢�����ӣ�����Ϊ�ݶ�grad������ɢ��div������������f�Ƕ��׿�΢��ʵ��������f��������˹���Ӷ���Ϊ��
    //----------------------------------------------------------------------------------  
	cv::Laplacian( g_srcImage, dst, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT );  
	cv::convertScaleAbs( dst, abs_grad_y );  
	cv::imshow("Laplacian", abs_grad_y); 
	

	//----------------------------------------------------------------------------------  
    //  scharrһ���Ҿ�ֱ�ӳ���Ϊ�˲��������������ӡ����������Ѿ�����������OpenCV����Ҫ�����Sobel���ӵ���������ڵ�,
    //----------------------------------------------------------------------------------  
	 //��3���� X�����ݶ�  
	cv::Scharr( g_srcImage, grad_x, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT );  
    cv:: convertScaleAbs( grad_x, abs_grad_x );  
    cv::imshow("��Ч��ͼ�� X����Scharr", abs_grad_x);   
	//��4����Y�����ݶ�  
	cv::Scharr( g_srcImage, grad_y, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT );  
    cv::convertScaleAbs( grad_y, abs_grad_y );  
    cv::imshow("��Ч��ͼ��Y����Scharr", abs_grad_y);

	cv::imshow( WINDOW_NAME1, g_srcImage );
}

//�ı�ͼ��ԱȶȺ�����ֵ�Ļص�����  
void contrastAndBright(int,void*){
	for(int y=0;y<g_srcImage.rows;y++){
		for(int x=0;x<g_srcImage.cols;x++){
			for(int c = 0; c < 3; c++ )  {
				g_resultImage.at<cv::Vec3b>(y,x)[c]= cv::saturate_cast<uchar>( (g_nContrastValue*0.01)*(g_srcImage.at<cv::Vec3b>(y,x)[c] ) + g_nBrightValue );  
			}
		}
	}
	 //��ʾͼ��  
    cv::imshow(WINDOW_NAME1, g_srcImage);  
    cv::imshow(WINDOW_NAME2, g_resultImage); 
}
//�ı�ͼ��ԱȶȺ�����ֵ��������  
void OpenCV_Function::contrastAndBrightByTrackbar(){
	system("color5F");   
	g_srcImage=cv::imread("girl-t1.jpg");

	g_resultImage= cv::Mat::zeros( g_srcImage.size(), g_srcImage.type());

	cv::namedWindow( WINDOW_NAME1, cv::WINDOW_AUTOSIZE );
	cv::namedWindow( WINDOW_NAME2, cv::WINDOW_AUTOSIZE );

	cv::createTrackbar("�Աȶȣ�", WINDOW_NAME2,&g_nContrastValue,300,contrastAndBright );  
    cv::createTrackbar("��   �ȣ�",WINDOW_NAME2,&g_nBrightValue,200,contrastAndBright );  

	contrastAndBright(g_nContrastValue,0);  
    contrastAndBright(g_nBrightValue,0);  


}
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

	g_srcImage = cv::imread( "girl-t1.jpg", 1 );
	g_templateImage = cv::imread( "girl-logo-t1.jpg", 1 );

	cv::namedWindow( WINDOW_NAME1, cv::WINDOW_AUTOSIZE );
	cv::namedWindow( WINDOW_NAME2, cv::WINDOW_AUTOSIZE );

	cv::createTrackbar( "����", WINDOW_NAME1, &g_nMatchMethod, g_nMaxTrackbarNum,on_Matching);
	on_Matching(0,0);
}

//�ָ�ͨ����Ȼ���ٻ�ϡ�
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