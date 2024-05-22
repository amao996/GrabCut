#include <iostream>
#include <opencv2/opencv.hpp>
#include "GCApplication.h"

using namespace std;
using namespace cv;

static void help()
{
	std::cout << "\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit\n"
		"\tr - restore the original image\n"
		"\tn - next ten iteration\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}


GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
	gcapp.mouseClick( event, x, y, flags, param );
}


int main()
{
	string filename = "I:\\test.jpg";
	Mat image = imread( filename, 1 );
	Size s;

	s.height = image.rows / 5;
	s.width = image.cols / 5;
	resize(image, image, s);

	help();

	const string winName = "image";
	namedWindow( winName, WINDOW_AUTOSIZE);
	setMouseCallback( winName, on_mouse, 0 );

	gcapp.setImageAndWinName( image, winName );
	gcapp.showImage();

	for(;;)
	{
		int c = cvWaitKey(0);
		switch( (char) c )
		{
		case '\x1b':	// exit
			cout << "Exiting ..." << endl;
			destroyWindow(winName);
			return 0;
		case 'r':	// restore
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':	// next iteration
			int iterCount = gcapp.getIterCount();
			int newIterCount;
			newIterCount = gcapp.nextIter();
			cout << "IterCount:" << newIterCount << endl;
			if( newIterCount > iterCount )
			{
				gcapp.showImage();
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}
	destroyAllWindows();
	return 0;
}