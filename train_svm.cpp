// TestDbProject.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include "sqlite3.h"
#include "stdafx.h"
#include <cstdlib>
#include "HEADER.h"

using namespace std;
using namespace cv;

int labelColors[10000][2] = { 0 };
int myClrs[TOTAL_COLOR_Nr + 1][3];
int dummy[10000][3];

vector<Point> contourArray;
vector<vector<Point>> imageContours;
vector<int> segmentClrs;

list<hsvc> col_hash_map;
coor c;

list<SegmentFeatures> featureVector;
SegmentFeatures mF;
list<SegmentMoments> momentVector;
SegmentMoments SegMom;

ImData myImData;
time_t tstart, tend, mstart, mend;

vector<int> lops, lops2;

stringstream imgstream, mystream, seq_stream, imgstr, segmentstream, filterstream;

//Static array
int label[pyrHeight][pyrWidth] = { 0 };
int I[pyrHeight][pyrWidth] = { 0 };
int Q[pyrHeight][pyrWidth + 1] = { 0 };

int EQ[MAX_EXP_NrOf_LABELS] = { 0 };
int relationLUT[TOTAL_COLOR_Nr + 1][TOTAL_COLOR_Nr + 1] = { 0 };
int labelNr, segment_no, label_no;

int centerSum[2][1] = { 0 };
int scalefactor; 

bool flag;
double sqrt6(double y)
{
	double x, z, tempf;
	unsigned long *tfptr = ((unsigned long *)&tempf) + 1;
	tempf = y;
	*tfptr = (0xbfcdd90a - *tfptr) >> 1;
	x = tempf;
	z = y*0.5;
	x = (1.5*x) - (x*x)*(x*z);    //The more you make replicates of this statement 
	//the higher the accuracy, here only 2 replicates are used  
	x = (1.5*x) - (x*x)*(x*z);
	return x*y;
}

//static int callback(void *NotUsed, int argc, char **argv, char **azColName){
//	int i;
//	for (i = 0; i<argc; i++){
//		printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
//	}
//	printf("\n");
//	return 0;
//}

//void saveFeatures(list<SegmentFeatures> featureVector, ImData &mid, int ivar) {
//	sqlite3 *db;
//	char *zErrMsg = 0;
//	int rc;
//	rc = sqlite3_open("turtlebotDb.db", &db);
//
//	for (int j = 0; j<mid.connComp.size(); j++) {
//
//		list<SegmentFeatures>::iterator it = featureVector.begin();
//		std::advance(it, j);
//
//		/* Create SQL statement */
//
//		stringstream ss;
//		//"INSERT INTO mynewqvector (id,ratio,e1,e2,segmentlocation,f0,f1,f2,f3,f4,hue) "
//		ss << "INSERT INTO night(id,ratio,e1,e2,segmentlocation,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,hue) "
//			<< "VALUES (" << ivar << myzero << j << "," << it->r << "," << it->elongation.at(0) << "," << it->elongation.at(1)
//			<< "," << it->segmentLocation << "," << it->AverageFilterOutputs.at(0) << "," << it->AverageFilterOutputs.at(1)
//			<< "," << it->AverageFilterOutputs.at(2) << "," << it->AverageFilterOutputs.at(3) << "," << it->AverageFilterOutputs.at(4)
//			<< "," << it->AverageFilterOutputs.at(5) << "," << it->AverageFilterOutputs.at(6) << "," << it->AverageFilterOutputs.at(7)
//			<< "," << it->AverageFilterOutputs.at(8) << "," << it->AverageFilterOutputs.at(9) << "," << it->AverageFilterOutputs.at(10)
//			<< "," << it->AverageFilterOutputs.at(11) << "," << it->AverageFilterOutputs.at(12) << "," << it->AverageFilterOutputs.at(13)
//			<< "," << it->AverageFilterOutputs.at(14) << "," << it->AverageFilterOutputs.at(15) << "," << it->AverageFilterOutputs.at(16)
//			<< "," << it->AverageFilterOutputs.at(17) << "," << it->AverageFilterOutputs.at(18) << "," << it->AverageFilterOutputs.at(19)
//			<< "," << it->AverageFilterOutputs.at(20) << "," << it->AverageFilterOutputs.at(21) << "," << it->AverageFilterOutputs.at(22)
//			<< "," << it->AverageFilterOutputs.at(23) << "," << it->AverageFilterOutputs.at(24) << "," << it->AverageFilterOutputs.at(25)
//			<< "," << it->AverageFilterOutputs.at(26) << "," << it->AverageFilterOutputs.at(27) << "," << it->AverageFilterOutputs.at(28)
//			<< "," << it->AverageFilterOutputs.at(29) << "," << it->AverageFilterOutputs.at(30) << "," << it->AverageFilterOutputs.at(31)
//			<< "," << it->AverageFilterOutputs.at(32) << "," << it->AverageFilterOutputs.at(33) << "," << it->AverageFilterOutputs.at(34)
//			<< "," << it->AverageFilterOutputs.at(35) << "," << it->AverageFilterOutputs.at(36) << "," << it->AverageFilterOutputs.at(37)
//			<< "," << it->AverageFilterOutputs.at(38) << "," << it->AverageFilterOutputs.at(39) << "," << it->AverageFilterOutputs.at(40)
//			<< "," << it->AverageFilterOutputs.at(41) << "," << it->AverageFilterOutputs.at(42) << "," << it->AverageFilterOutputs.at(43)
//			<< "," << it->AverageFilterOutputs.at(44) << "," << it->AverageFilterOutputs.at(45) << "," << it->AverageFilterOutputs.at(46)
//			<< "," << it->AverageFilterOutputs.at(47) << "," << it->AverageFilterOutputs.at(48) << "," << it->AverageFilterOutputs.at(49)
//			<< "," << it->AverageFilterOutputs.at(50) << "," << it->AverageFilterOutputs.at(51) << "," << it->AverageFilterOutputs.at(52)
//			<< "," << it->AverageFilterOutputs.at(53) << "," << it->AverageFilterOutputs.at(54) <<	 /*"," << it->elongation.at(1) / it->elongation.at(0) <<*/  ");";
//
//		const string temp = ss.str();
//		const char *sql = temp.c_str();
//
//		std::cout << std::endl << sql << std::endl << std::endl << std::endl;
//
//		/* Execute SQL statement */
//		rc = sqlite3_exec(db, sql, callback, 0, &zErrMsg);
//		if (rc != SQLITE_OK){
//			fprintf(stderr, "SQL error: %s\n", zErrMsg);
//			sqlite3_free(zErrMsg);
//
//			system("pause");
//			waitKey(50);
//		}
//		else{
//			fprintf(stdout, "Table created successfully\n");
//		}
//
//	}
//
//
//	sqlite3_close(db);
//
//}

void LUT(int relationLUT[TOTAL_COLOR_Nr + 1][TOTAL_COLOR_Nr + 1]){


	for (int t = 0; t <TOTAL_COLOR_Nr + 1; t++){
		relationLUT[t][t] = 1;
	}
	relationLUT[24][25] = 1;//mx white & white
	relationLUT[25][24] = 1;//white & mx white 
	//relationLUT[23][22]=1;//black & mx black
	//relationLUT[22][23]=1;
	relationLUT[23][18] = 1;//black & dark gray
	relationLUT[18][23] = 1;
	relationLUT[18][22] = 1;//dark gray & mx black
	relationLUT[22][18] = 1;
	//relationLUT[24][19]=1;//mx white & light gray 
	relationLUT[8][9] = 1; // light blue & dark blue  
	relationLUT[21][3] = 1;//yellow & brown
	relationLUT[3][20] = 1;

}

void myColors(){
	int twentyfivecolors[TOTAL_COLOR_Nr + 1][3] = {
		{ 30, 30, 30 },
		{ 255, 0, 0 },//red				1
		{ 255, 128, 0 },//orange		2
		{ 255, 255, 0 },//yellow		3
		{ 219, 253, 0 },//yellowgreen	4
		{ 102, 255, 102 },//lightgreen	5
		{ 0, 51, 0 },//darkgreen		6
		{ 0, 255, 255 },//cyan			7
		{ 153, 204, 255 },//lightblue	8
		{ 0, 0, 204 },//darkblue		9
		{ 153, 51, 255 },//violet		10
		{ 76, 0, 153 },//purple			11
		{ 149, 0, 104 },//pinkpurple	12
		{ 255, 122, 102 },//pink1(yavruagzi) 13
		{ 199, 40, 63 },//pink2(cherry)	14
		{ 255, 204, 255 },//pink3(lightpink) 15
		{ 255, 0, 255 },//magenta		16
		{ 255, 70, 255 },//pink			17
		{ 64, 64, 64 },//darkgray		18
		{ 192, 192, 192 },//lightgray	19
		{ 58, 25, 25 },//brown1(taba)	20
		{ 45, 21, 0 },//brown			21
		{ 47, 42, 42 },//mixedblack		22
		{ 0, 0, 0 },//black				23
		{ 240, 240, 240 },//mixedwhite	24
		{ 255, 255, 255 },//white		25
	};

	for (int i = 0; i<TOTAL_COLOR_Nr + 1; i++){
		for (int j = 0; j<3; j++){

			myClrs[i][j] = twentyfivecolors[i][j];

		}
	}

}

void colorImg(vector<Mat> channels, Mat ch0, Mat ch1, Mat ch2, int I[pyrHeight][pyrWidth], int imgnr){
	channels.clear();

	myColors();

	for (int colorindex = 1; colorindex<TOTAL_COLOR_Nr + 1; colorindex++)
	{
		for (int i = 0; i < myImData.h; i++) {
			for (int j = 0; j < myImData.w; j++) {

				int colorval = I[i][j];

				if (colorval == colorindex){
					ch2.at<uchar>(i, j) = myClrs[colorindex][0];
					ch1.at<uchar>(i, j) = myClrs[colorindex][1];
					ch0.at<uchar>(i, j) = myClrs[colorindex][2];
				}

			}
		}
	}
	Mat fin_img2;
	channels.push_back(ch0); //b
	channels.push_back(ch1); //g
	channels.push_back(ch2); //r
	merge(channels, fin_img2);

	//imgstr << "colored_by_respected_colors_"<< imagename << quant << type;
	imgstr << imagename1 << "_" << imgnr << quant << type;
	string myfilename = imgstr.str();
	imgstr.str("");
	imwrite(myfilename, fin_img2);

	ch0.release();
	ch1.release();
	ch2.release();
	fin_img2.release();
	channels.clear();
}

void Labeling(int &labelNr, int label[pyrHeight][pyrWidth], int I[pyrHeight][pyrWidth], int Q[pyrHeight][pyrWidth + 1], int EQ[MAX_EXP_NrOf_LABELS], ImData &myImData){
	// Label (0,0) start point
	int L = 0;
	labelNr = 0;
	++L; ++labelNr;
	EQ[L] = (L);
	label[0][0] = L; Q[0][1] = L;

	// Label first row 	
	for (int x = 1; x<myImData.w; x++){

		int y = 0;
		int n1x = x - 1;

		if (I[y][n1x] == I[y][x]){
			label[y][x] = label[y][n1x];
			Q[y][x + 1] = label[y][x];
		}
		if (I[y][n1x] != I[y][x]){
			++L; ++labelNr;
			EQ[L] = (L);
			label[y][x] = L;
			Q[y][x + 1] = L;
		}
	}

	// Label first column starting from second row	

	for (int y = 1; y<myImData.h; y++){
		if (I[y][0] == I[y - 1][0]){
			label[y][0] = label[y - 1][0];
			Q[y][1] = label[y][0];
		}

		if (I[y][0] != I[y - 1][0]){
			++L; ++labelNr;
			EQ[L] = (L);
			label[y][0] = L;
			Q[y][1] = label[y][0];
		}
	}

	//Label the rest of the img

	for (int x = 1; x<myImData.w; x++){
		for (int y = 1; y<myImData.h; y++){


			int sx = x - 1; int sy = y;
			int tx = x;	 int ty = y - 1;

			if (I[y][x] == I[sy][sx] && I[y][x] != I[ty][tx]){
				label[y][x] = label[sy][sx];
			}

			if (I[y][x] != I[sy][sx] && I[y][x] == I[ty][tx]){
				label[y][x] = label[ty][tx];
			}

			if (I[y][x] != I[sy][sx] && I[y][x] != I[ty][tx]){
				++L; ++labelNr;
				EQ[L] = (L);
				label[y][x] = L;
			}

			if (I[y][x] == I[sy][sx] && I[y][x] == I[ty][tx] && label[sy][sx] == label[ty][tx]){
				label[y][x] = label[ty][tx];
			}

			if (I[y][x] == I[sy][sx] && I[y][x] == I[ty][tx] && label[sy][sx] != label[ty][tx]){
				int comp = (label[sy][sx]<label[ty][tx]); // Ls < Lt -->  1
				// Ls > Lt -->  0
				int L1, L2; //L1<L2
				comp ? L1 = label[sy][sx] : L1 = label[ty][tx];
				comp ? L2 = label[ty][tx] : L2 = label[sy][sx];

				label[y][x] = L1;
				EQ[L2] = L1;
			}
			Q[y][x + 1] = label[y][x];

		}
	}
	for (int i = 0; i<myImData.h; i++){
		Q[i][0] = label[i][1];

	}

}

void LabelEqualization(int EQ[MAX_EXP_NrOf_LABELS], int label[pyrHeight][pyrWidth], ImData &myImData, int labelColors[10000][2]){
	//Equalization of labels
	for (int k = 1; k<MAX_EXP_NrOf_LABELS; k++){

		if (EQ[k] == 0){ break; }

		if (EQ[k] != k){
			EQ[EQ[k]] == EQ[k] ? 1 : EQ[k] = EQ[EQ[EQ[k]]];
		}

		for (int i = 0; i < myImData.h; i++) {
			for (int j = 0; j < myImData.w; j++) {
				if (label[i][j] == k)
					label[i][j] = EQ[k];
				Q[i][j + 1] = label[i][j];

				//labelColors[label[i][j]][0] = I[i][j];
			}
		}
	}
}

void createHash(String dy){

	string line;
	ifstream myfile(dy);

	while (getline(myfile, line))
	{
		stringstream   linestream(line);
		string         data;
		int hl, hh, sl, sh, vl, vh, color_name;// HueLow, HueHigh, SaturationLow, SaturationHigh, ValueLow, ValueHigh

		getline(linestream, data, '\t');

		linestream >> hh >> sl >> sh >> vl >> vh >> color_name;
		hsvc new_hsvc;
		new_hsvc.hlow = atoi(data.c_str());
		new_hsvc.hhigh = hh;
		new_hsvc.slow = sl;
		new_hsvc.shigh = sh;
		new_hsvc.vlow = vl;
		new_hsvc.vhigh = vh;
		new_hsvc.col_name = color_name;

		col_hash_map.push_back(new_hsvc);

	}
	myfile.close();

}

void keepcolors(String clrs){
	int cidx = 0;
	string line;
	ifstream file(clrs);

	while (getline(file, line))
	{
		stringstream   linestream(line);
		string        data;
		int r, g, b;

		getline(linestream, data, '\t');
		linestream >> g >> b;

		/* Array implementation*/
		dummy[cidx][0] = atoi(line.c_str());
		dummy[cidx][1] = g;
		dummy[cidx][2] = b;

		++cidx;

	}
	file.close();

}

Mat readFilter(int filtreSirasi, int filterSize)
{

	Mat filterOrg = Mat(filterSize, filterSize, CV_32FC1);
	Mat filter;

	// QFile file(dirr);
	char filtreDosya[50];
	sprintf_s(filtreDosya, "filtreler\\filtre%i.txt", filtreSirasi);
	fstream file_op(filtreDosya, ios::in);

	for (int j = 0; j<filterSize; j++)
		for (int k = 0; k<filterSize; k++){
			file_op >> filterOrg.at<float>(k, j);
		}

	file_op.close();

	convertScaleAbs(filterOrg, filter, 128, 128);

	Mat resizedFilter;

	resize(filter, resizedFilter, resizedFilter.size(), 5, 5);

	//namedWindow("filter");

	//cv::imshow("filter",resizedFilter);

	//waitKey();

	//destroyWindow("filter");


	return filterOrg;
}

void applyFilter(ImData &mid, Mat filter_out, int index) //this function gets the address of ImData and routes filters directly into filtered outputs in this struct.
{
	Mat result = Mat::zeros(mid.h, mid.w, CV_8UC1);
	mid.filter.push_back(Mat::zeros(mid.h, mid.w, CV_8UC1));

	GaussianBlur(mid.intensity, mid.filter[index], Size(5, 5), 5, 5);

	filter2D(mid.filter.at(index), result, result.depth(), filter_out);

	mid.filter[index] = result;
}

void features(ImData &mid, int seg_no){

	//	int segmentNr=0;
	//for (int k = 0; k< mid.connComp.size(); k++){

		int k = seg_no;
		int cx = 0;
		int cy = 0;

		
		double filtSum54 = 0; //hue

		int centerxSum = 0;
		int centerySum = 0;

		//second order moments
		double m20 = 0;
		double m02 = 0;
		double m11 = 0;

		double A = mid.connComp.at(k).size(); // Area of the segment = m00
		//	SegMom.m00 = A;

		mF.r = A / (pyrHeight* pyrWidth);

		for (int s = 0; s < mid.connComp.at(k).size(); s++)
		{
			// center of mass calculation --> mass(cx,cy)

			//centerSum = centerSum + memberpixel;
			int memberx = mid.connComp.at(k).at(s).x;
			int membery = mid.connComp.at(k).at(s).y;

			centerxSum = centerxSum + memberx;
			centerySum = centerySum + membery;
			if (s == mid.connComp.at(k).size() - 1)
			{
				//cx and cy calculation
				cx = (centerxSum) / A;
				cy = (centerySum) / A;

			}
		}

		if (cy <= UpObjTh){ mF.segmentLocation = 0.9; }
		else if (cy>UpObjTh && cy <= MidObjTh){ mF.segmentLocation = 0.6; }
		else if (cy>MidObjTh){ mF.segmentLocation = 0.3; }

		for (int s = 0; s < mid.connComp.at(k).size(); s++){

			m20 = m20 + ((mid.connComp.at(k).at(s).x - cx)*(mid.connComp.at(k).at(s).x - cx));
			m02 = m02 + ((mid.connComp.at(k).at(s).y - cy)*(mid.connComp.at(k).at(s).y - cy));
			m11 = m11 + ((mid.connComp.at(k).at(s).x - cx)*(mid.connComp.at(k).at(s).y - cy));


			double	dumVal54 = mid.filter.at(numberoffilters).at<uchar>(mid.connComp.at(k).at(s).y, mid.connComp.at(k).at(s).x);
			filtSum54 = filtSum54 + dumVal54;

			if (s == mid.connComp.at(k).size() - 1)
			{

				//hue normalization is different
				double f54 = filtSum54 / A;
				f54 = f54 / 180;
				//mF.AverageFilterOutputs.push_back(f54);
				mF.hueval = f54;
				//mF.mass = Point(cx,cy);

				//calculation of major and minor axis

				m20 = m20 / A; SegMom.m20 = m20;
				m02 = m02 / A; SegMom.m02 = m02;
				m11 = m11 / A; SegMom.m11 = m11;


				double k4 = sqrt6(abs((4 * (m11)*(m11)) - ((m20 - m02)*(m20 - m02))));

				double lambda1 = sqrt6(((m20 + m02) / 2) + k4); //major
				lambda1 = lambda1 / hypo;	// = q2 (normalized major length)
				double lambda2 = sqrt6(abs(((m20 + m02) / 2) - k4)); //minor
				lambda2 = lambda2 / hypo;	// =  q3 (normalized minor length)

				mF.shaperatio = lambda2 / lambda1;
				//mF.elongation.push_back(lambda1);
				//mF.elongation.push_back(lambda2);
				//eccentricity = abs((((m20 - m02)*(m20 - m02))-(4*(m11)*(m11)))/((m20 + m02)*(m20 + m02)));

				//featureVector.push_back(mF);
				//	momentVector.push_back(SegMom);

				
				//mF.AverageFilterOutputs.clear();
				//mF.elongation.clear();

			}
		}
	//}
}

float sqrt5(const float m)
{
	float i = 0;
	float x1, x2;
	while ((i*i) <= m)
		i += 0.1f;
	x1 = i;
	for (int j = 0; j<10; j++)
	{
		x2 = m;
		x2 /= x1;
		x2 += x1;
		x2 /= 2;
		x1 = x2;
	}
	return x2;
}

float labels_array[250] = { 0 };
float trainingData[250][numberofbins] = { 0 };
float samples[1000][numberofbins] = { 0 };
int imgandsegnr[1000][2]={0};
getdataclass myData;

void hesapla(int, int, int);
void testyap(int, int, int);

void hesapla2(int, int, int);
void testyap2(int, int);

vector<int> segmentsec(vector<vector<Point>>);

int index = 0 ;
int sindex = 0;
int train_sample_number,  train_sample_number2, test_sample_number2, test_image_number;
int test_sample_number = 0;
Mat filt_crop;
//int histSize = 3;    // bin size
float range[] = { 0, 255 };
const float *ranges[] = { range };
int main()
{	//tstart = time(0);

	LUT(relationLUT);
	//I defined hypotenus lenght of the image in MyLibConstants. It is equal to 400 for 240*320 image. 
	//If image size changes, new hypotenus can be calculated by: 
	//int hypotenus = sqrt5((pyrHeight * pyrHeight) + ( pyrWidth * pyrWidth));
	keepcolors("SegmentColors.txt");
	createHash("ColorQuantas.txt");

	myData.getTrainData("train.txt");
	myData.getTestData("test.txt");

	char choice1;
	cout << "want to train svm? (Y//N)" << endl;
	cin >> choice1;


	if (choice1 == 'y' || choice1 == 'Y'){
	
	char choice2;
	cout << "want to use segment location? (Y//N)" << endl;
	cin >> choice2;

	if (choice2 == 'y' || choice2 == 'Y'){
		cout << "Number of train samples:" << endl;
		cin >> train_sample_number;
		do{
			hesapla(myData.trainArr[index][0], myData.trainArr[index][1], myData.labelsArr[index]);
			index = index + 1;


		} while (index<train_sample_number);



		Mat labelsMat(train_sample_number, 1, CV_32FC1, labels_array);
		Mat trainingDataMat(train_sample_number, numberofbins, CV_32FC1, trainingData);


		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		// Train the SVM
		CvSVM SVM;
		SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
		//svm.save( "my_svm.xml" ); // to save in XML
		SVM.save("my_svm_3.xml");
		segmentClrs.clear();
		contourArray.clear();

		mF.AverageFilterOutputs.clear();

		myImData.filter.clear();
		myImData.hsv_filter.clear();

		myImData.connComp.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		labelNr = 0;


		cout << "Number of test samples:" << endl;
		cin >> test_sample_number;
		do{
			testyap(myData.testArr[sindex][0], myData.testArr[sindex][1],test_sample_number);
			sindex = sindex + 1;

		} while (sindex<test_sample_number);

		Mat testing_data = Mat(test_sample_number, numberofbins, CV_32FC1, samples);
		Mat testing_classifications = Mat(test_sample_number, 1, CV_32FC1);
		SVM.predict(testing_data, testing_classifications);
		cout << testing_classifications << endl;
	}

	else {
		cout << "Number of train samples:" << endl;
		cin >> train_sample_number2;
		do{
			hesapla2(myData.trainArr[index][0], myData.trainArr[index][1], myData.labelsArr[index]);
			index = index + 1;


		} while (index<train_sample_number2);



		Mat labelsMat(train_sample_number2, 1, CV_32FC1, labels_array);
		Mat trainingDataMat(train_sample_number2, numberofbins, CV_32FC1, trainingData);


		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
		// Train the SVM
		CvSVM SVM;
		SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

		segmentClrs.clear();
		contourArray.clear();

		mF.AverageFilterOutputs.clear();

		myImData.filter.clear();
		myImData.hsv_filter.clear();

		myImData.connComp.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		labelNr = 0;


		cout << "Number of test samples:" << endl;
		cin >> test_sample_number2;
		do{
			testyap2(myData.testArr[sindex][0], myData.testArr[sindex][1]);
			sindex = sindex + 1;

		} while (sindex<test_sample_number2);

		Mat testing_data = Mat(test_sample_number2, numberofbins, CV_32FC1, samples);
		Mat testing_classifications = Mat(test_sample_number2, 1, CV_32FC1);
		SVM.predict(testing_data, testing_classifications);
		cout << testing_classifications << endl;
	}

	
	}
		//------ test without train -------//
			else{
		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		CvSVM SVM;
		SVM.load("my_svm.xml");
		
		
		cout << "Number of test images:" << endl;
		cin >> test_image_number;
			test_sample_number = 0;

	
		while (sindex<test_image_number){
			
					testyap(myData.testArr[sindex][0],1,test_sample_number);
					sindex = sindex + 1;
		}

		Mat testing_data = Mat(test_sample_number, numberofbins, CV_32FC1, samples);
		Mat testing_classifications = Mat(test_sample_number, 1, CV_32FC1);
		SVM.predict(testing_data, testing_classifications);
		cout << testing_classifications << endl;
			}

 	waitKey(0);
	return 0;
}

void hesapla(int image_no, int segment_no, int label_no){
		MatND hist;
		labels_array[index] = label_no;
		

		int ivar = image_no;
		
		mystream << imagename1 << std::setfill('0') << std::setw(4) << ivar << type;
		string myfilename = mystream.str();
		mystream.str("");
		Mat src = imread(myfilename, CV_LOAD_IMAGE_COLOR);

		Mat dst;
		pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));
		bilateralFilter(dst, myImData.original, 9, 30, 30);

		cvtColor(myImData.original, myImData.intensity, CV_BGR2GRAY);
		cvtColor(myImData.original, myImData.hsvImg, CV_BGR2HSV);

		myImData.h = myImData.original.rows;
		myImData.w = myImData.original.cols;

		vector<Mat> hsvchannels;
		split(myImData.hsvImg, hsvchannels); 

		myImData.hsv_filter.push_back(hsvchannels[0]);
		myImData.hsv_filter.push_back(hsvchannels[1]);
		myImData.hsv_filter.push_back(hsvchannels[2]);


		hsvchannels.clear();



		//keep filters in here
		vector<Mat> filters;

		for (int f = 0; f<numberoffilters; f++)
		{
			filters.push_back(readFilter(f, 29));
		}
		for (int i = 0; i<numberoffilters; i++){
			//filter original image and create filtered outputs
			applyFilter(myImData, filters[i], i);

			
		}
		myImData.filter.push_back(myImData.hsv_filter.at(0)); //Hue filter result added to 5 spatial filter result
		filters.clear();
		//int count[TOTAL_COLOR_Nr];
		//std::fill( count, count+TOTAL_COLOR_Nr, 0);



		for (int k = 0; k < (myImData.h * myImData.w); k++){
			int x = k % (myImData.w);
			int y = (k - x) % (myImData.w - 1);
			bool flag = false;

			for (list<hsvc>::iterator it = col_hash_map.begin(); it != col_hash_map.end(); ++it){


				int val_h = myImData.hsv_filter.at(0).at<uchar>(y, x);
				int val_s = myImData.hsv_filter.at(1).at<uchar>(y, x);
				int val_v = myImData.hsv_filter.at(2).at<uchar>(y, x);


				if (val_h >= it->hlow && val_h <= it->hhigh && val_s >= it->slow &&
					val_s <= it->shigh && val_v >= it->vlow && val_v <= it->vhigh){
					I[y][x] = it->col_name;

					flag = true; break;
				}

			}
		}

		Labeling(labelNr, label, I, Q, EQ, myImData);

		LabelEqualization(EQ, label, myImData, labelColors);


		/*Merge small components with their nearest component*/

		std::unordered_map<int, int> occurrences;

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				++occurrences[label[i][j]];
			}
		}

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				if (occurrences[label[i][j]] < MAX_PxNr_SMALL_AREA) {
					EQ[label[i][j]] = Q[i][j];
					Q[i][j + 1] = Q[i][j];

				}
			}
		}
		occurrences.clear();

		// LabelEqualization(EQ, label, myImData,labelColors);

		vector<int> nIndx;
		int indx = 1;

		while (indx != labelNr + 1){
			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int val = label[i][j];
					if (val == indx){
						contourArray.push_back(Point(j, i));
					}
				}
			}
			if (contourArray.empty() == false){

				myImData.connComp.push_back(contourArray);
				int clrv;

				// int clrv = I[contourArray.at(0).y][contourArray.at(0).x]; //yanlýþ renk birleþtirme olmasýn diye deðiþtiriyorum burayý 07.10.2015
				if (contourArray.size() == 1){
					clrv = I[contourArray.at(0).y][contourArray.at(0).x];
				}
				if (contourArray.size()>1){
					clrv = I[contourArray.at(contourArray.size() - 1).y][contourArray.at(contourArray.size() - 1).x];
				}

				segmentClrs.push_back(clrv);

				if (contourArray.size()> MIN_PxNr_BIG_AREA){
					nIndx.push_back(myImData.connComp.size() - 1);
				}

			}
			++indx;
		}

		for (int nfc = 0; nfc< nIndx.size(); nfc++){

			int numberofcomponents = nIndx.at(nfc);
			Mat component_Img = Mat::zeros(myImData.h, myImData.w, CV_8UC1);
			Mat dilated_component_Img, dst;
			Mat eroded_;
			// Create binary image of big segment
			for (int comp = 0; comp < myImData.connComp.at(numberofcomponents).size(); comp++){
				Point component = myImData.connComp.at(numberofcomponents).at(comp);
				component_Img.at<uchar>(component.y, component.x) = 255;
			}

			dilate(component_Img, dilated_component_Img, dilation_element);

			// Obtain adjacent parts
			cv::bitwise_xor(component_Img, dilated_component_Img, dst);
			//imshow("dst", dst); //adjacent img
			//	waitKey(0);


			vector<Point> nonZeroCoordinates;		//keep adjacent pixels in here
			findNonZero(dst, nonZeroCoordinates);

			int	ColorNr1 = segmentClrs.at(numberofcomponents);
			int ColorNr2;
			int newLabel = label[myImData.connComp.at(numberofcomponents).at(0).y][myImData.connComp.at(numberofcomponents).at(0).x];

			for (int g = 0; g<nonZeroCoordinates.size(); g++){

				Point AdjPoint = nonZeroCoordinates.at(g);
				ColorNr2 = I[AdjPoint.y][AdjPoint.x];
				if (relationLUT[ColorNr1][ColorNr2] == 1){

					EQ[label[AdjPoint.y][AdjPoint.x]] = newLabel;

				}

			}

		}

		LabelEqualization(EQ, label, myImData, labelColors);
		myImData.connComp.clear();

		int nindx = 1;
		while (nindx != labelNr + 1){

			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int nval = label[i][j];
					if (nval == nindx){

						contourArray.push_back(Point(j, i));

						/*ch2.at<uchar>(i, j) = dummy[(nindx)][0];
						ch1.at<uchar>(i, j) = dummy[(nindx)][1];
						ch0.at<uchar>(i, j) = dummy[(nindx)][2];*/

					}
				}
			}
			if (contourArray.empty() == false){
				myImData.connComp.push_back(contourArray);
			}
			++nindx;
		}



		 scalefactor = myImData.connComp.at(segment_no).size();
			features(myImData,segment_no);		
		//	cout << mF.hueval << endl << mF.shaperatio << endl;

		//	saveFeatures(featureVector,myImData,ivar);

			Mat goster_filtre(myImData.h, myImData.w, CV_8UC1, Scalar(background));

			vector<vector<float>> svminput;
			vector<float> binvector;
			//int histSize = 48;    // bin size
			//float range[] = { 0, 255 };
			//const float *ranges[] = { range };
			
			
			for (int filter_no = 0; filter_no < numberoffilters; filter_no++)
			
			{

				for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

				{
					Point component = myImData.connComp.at(segment_no).at(d);

					filt_crop.push_back(myImData.filter.at(filter_no).at<uchar>(component.y, component.x));
				//	goster_filtre.at<uchar>(component.y, component.x) = myImData.filter.at(filter_no).at<uchar>(component.y, component.x);
					
				}

			// Calculate histogram
		
			calcHist(&filt_crop, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
						
			// Show the calculated histogram in command window
			/*double total;
			total = filt_crop.rows * filt_crop.cols;*/
			for (int h = 0; h < histSize; h++)
			{
				float binVal = hist.at<float>(h);
				//cout<<" "<<binVal;
				binvector.push_back(binVal / (scalefactor));
				//	trainingData[index][h] = binVal;

			}
			filt_crop.release();
			hist.release();
			}

			std::unordered_map<int, int> voting;
			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

				{
					Point component = myImData.connComp.at(segment_no).at(d);

					++voting[I[component.y][component.x]];
					
				}

				int currentMax = 0;
				int arg_max = 0;
				for(auto it = voting.cbegin(); it != voting.cend(); ++it ) {
					if (it ->second > currentMax) {
						arg_max = it->first;
						currentMax = it->second;
					}
				}
				//cout << "Value " << arg_max << " occurs " << currentMax << " times " << endl;					
				
			float arg_max_f = arg_max/25;
				
			binvector.push_back(mF.hueval);
			binvector.push_back(mF.shaperatio);
			binvector.push_back(mF.segmentLocation);
			binvector.push_back(mF.r);
			binvector.push_back(arg_max_f);

				
		
			for (int s = 0; s < numberofbins; s++){
				trainingData[index][s] = binvector.at(s);
			}

			//482 svm input saved to binvector for an instance
			
 			svminput.push_back(binvector);

		voting.clear();
		nindx = 0;
		nIndx.clear();
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth; pw++){
				label[ph][pw] = 0;
				I[ph][pw] = 0;

			}
		}
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth + 1; pw++){

				Q[ph][pw] = 0;

			}
		}

		for (int eqn = 0; eqn<MAX_EXP_NrOf_LABELS; eqn++){ EQ[eqn] = 0; }

	
		filt_crop.release();
		binvector.clear();
		segmentClrs.clear();
		contourArray.clear();
		//mychannels.clear();
		//fin_img.release();
		//featureVector.clear();
		//ch0.release();
		//ch1.release();
		//ch2.release();
		mF.AverageFilterOutputs.clear();
		
		myImData.filter.clear();
		myImData.hsv_filter.clear();
		//	myImData.hsv_col_info.clear();
		myImData.connComp.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		labelNr = 0;
	}

void testyap(int image_no, int segment_nos, int test_sample_number){
		MatND hist;
		int ivar = image_no;

		mystream << imagename1 << std::setfill('0') << std::setw(4) << ivar << type;
		string myfilename = mystream.str();
		mystream.str("");
		Mat src = imread(myfilename, CV_LOAD_IMAGE_COLOR);

		Mat dst;
		pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));
		bilateralFilter(dst, myImData.original, 9, 30, 30);

		cvtColor(myImData.original, myImData.intensity, CV_BGR2GRAY);
		cvtColor(myImData.original, myImData.hsvImg, CV_BGR2HSV);

		myImData.h = myImData.original.rows;
		myImData.w = myImData.original.cols;

		vector<Mat> hsvchannels;
		split(myImData.hsvImg, hsvchannels);

		myImData.hsv_filter.push_back(hsvchannels[0]);
		myImData.hsv_filter.push_back(hsvchannels[1]);
		myImData.hsv_filter.push_back(hsvchannels[2]);

		hsvchannels.clear();

		//keep filters in here
		vector<Mat> filters;

		for (int f = 0; f<numberoffilters; f++)
		{
			filters.push_back(readFilter(f, 29));
		}
		for (int i = 0; i<numberoffilters; i++){
			//filter original image and create filtered outputs
			applyFilter(myImData, filters[i], i);

			
		}
		myImData.filter.push_back(myImData.hsv_filter.at(0)); //Hue filter result added to 5 spatial filter result
		filters.clear();

		for (int k = 0; k < (myImData.h * myImData.w); k++){
			int x = k % (myImData.w);
			int y = (k - x) % (myImData.w - 1);
			bool flag = false;

			for (list<hsvc>::iterator it = col_hash_map.begin(); it != col_hash_map.end(); ++it){

				int val_h = myImData.hsv_filter.at(0).at<uchar>(y, x);
				int val_s = myImData.hsv_filter.at(1).at<uchar>(y, x);
				int val_v = myImData.hsv_filter.at(2).at<uchar>(y, x);

				if (val_h >= it->hlow && val_h <= it->hhigh && val_s >= it->slow &&
					val_s <= it->shigh && val_v >= it->vlow && val_v <= it->vhigh){

				I[y][x] = it->col_name;

					flag = true; break;
				}

			}
		}

		Labeling(labelNr, label, I, Q, EQ, myImData);

		LabelEqualization(EQ, label, myImData, labelColors);

		/*Merge small components with their nearest component*/
		std::unordered_map<int, int> occurrences;

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				++occurrences[label[i][j]];
			}
		}

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				if (occurrences[label[i][j]] < MAX_PxNr_SMALL_AREA) {
					EQ[label[i][j]] = Q[i][j];
					Q[i][j + 1] = Q[i][j];

				}
			}
		}
		occurrences.clear();

		// LabelEqualization(EQ, label, myImData,labelColors);
		vector<int> nIndx;
		int indx = 1;

		while (indx != labelNr + 1){
			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int val = label[i][j];
					if (val == indx){
						contourArray.push_back(Point(j, i));
					}
				}
			}
			if (contourArray.empty() == false){

				myImData.connComp.push_back(contourArray);
				int clrv;

				if (contourArray.size() == 1){
					clrv = I[contourArray.at(0).y][contourArray.at(0).x];
				}
				if (contourArray.size()>1){
					clrv = I[contourArray.at(contourArray.size() - 1).y][contourArray.at(contourArray.size() - 1).x];
				}

				segmentClrs.push_back(clrv);

				if (contourArray.size()> MIN_PxNr_BIG_AREA){
					nIndx.push_back(myImData.connComp.size() - 1);
				}

			}
			++indx;
		}

		for (int nfc = 0; nfc< nIndx.size(); nfc++){

			int numberofcomponents = nIndx.at(nfc);
			Mat component_Img = Mat::zeros(myImData.h, myImData.w, CV_8UC1);
			Mat dilated_component_Img, dst;
			Mat eroded_;
			// Create binary image of big segment
			for (int comp = 0; comp < myImData.connComp.at(numberofcomponents).size(); comp++){
				Point component = myImData.connComp.at(numberofcomponents).at(comp);
				component_Img.at<uchar>(component.y, component.x) = 255;
			}

			dilate(component_Img, dilated_component_Img, dilation_element);

			// Obtain adjacent parts
			cv::bitwise_xor(component_Img, dilated_component_Img, dst);
	
			vector<Point> nonZeroCoordinates;		//keep adjacent pixels in here
			findNonZero(dst, nonZeroCoordinates);

			int	ColorNr1 = segmentClrs.at(numberofcomponents);
			int ColorNr2;
			int newLabel = label[myImData.connComp.at(numberofcomponents).at(0).y][myImData.connComp.at(numberofcomponents).at(0).x];

			for (int g = 0; g<nonZeroCoordinates.size(); g++){

				Point AdjPoint = nonZeroCoordinates.at(g);
				ColorNr2 = I[AdjPoint.y][AdjPoint.x];
				if (relationLUT[ColorNr1][ColorNr2] == 1){

					EQ[label[AdjPoint.y][AdjPoint.x]] = newLabel;

				}

			}

		}

		LabelEqualization(EQ, label, myImData, labelColors);

		myImData.connComp.clear();

		int nindx = 1;
		while (nindx != labelNr + 1){

			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int nval = label[i][j];
					if (nval == nindx){

						contourArray.push_back(Point(j, i));

					}
				}
			}
			if (contourArray.empty() == false){
				myImData.connComp.push_back(contourArray);
			}
			++nindx;
		}

			vector<int> pruneseg;
				segmentsec(myImData.connComp);
				pruneseg = lops;

			if (pruneseg.empty() == false){	

			for (int ls = 0; ls < lops.size() ; ls++)
			
{
				segment_no =  pruneseg.back();
				pruneseg.pop_back();

		scalefactor = myImData.connComp.at(segment_no).size();

		features(myImData, segment_no);
		vector<float> binvector;
	
	
		for (int filter_no = 0; filter_no < numberoffilters; filter_no++)

		{

			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

			{
				Point component = myImData.connComp.at(segment_no).at(d);

				filt_crop.push_back(myImData.filter.at(filter_no).at<uchar>(component.y, component.x));
				//	goster_filtre.at<uchar>(component.y, component.x) = myImData.filter.at(filter_no).at<uchar>(component.y, component.x);
	

			}

			// Calculate histogram
			
			calcHist(&filt_crop, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

			for (int h = 0; h < histSize; h++)
			{
				float binVal = hist.at<float>(h);
				binvector.push_back(binVal / (scalefactor));
			}
			filt_crop.release();
			hist.release();
		}

			//	std::unordered_map<int, int> voting;
			//for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

			//	{
			//		Point component = myImData.connComp.at(segment_no).at(d);
			//		++voting[I[component.y][component.x]];
			//		
			//	}

			//	int currentMax = 0;
			//	int arg_max = 0;
			//	for(auto it = voting.cbegin(); it != voting.cend(); ++it ) {
			//		if (it ->second > currentMax) {
			//			arg_max = it->first;
			//			currentMax = it->second;
			//		}
			//	}
			//	//cout << "Value " << arg_max << " occurs " << currentMax << " times " << endl;					
			//	
			//float arg_max_f = arg_max/26;
				
		binvector.push_back(mF.hueval);
		binvector.push_back(mF.shaperatio);
		binvector.push_back(mF.segmentLocation);
		binvector.push_back(mF.r);
		//binvector.push_back(arg_max_f);

		for (int s = 0; s < numberofbins; s++){
			samples[test_sample_number][s] = binvector.at(s);
		}

			++test_sample_number;
			
		filt_crop.release();
		binvector.clear();

		mF.AverageFilterOutputs.clear();		
		
}
		}
		labelNr = 0;
		segmentClrs.clear();
		contourArray.clear();

		myImData.connComp.clear();
		nindx = 0;
		nIndx.clear();
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth; pw++){
				label[ph][pw] = 0;
				I[ph][pw] = 0;

			}
		}
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth + 1; pw++){

				Q[ph][pw] = 0;

			}
		}

		for (int eqn = 0; eqn<MAX_EXP_NrOf_LABELS; eqn++){ EQ[eqn] = 0; }
		myImData.filter.clear();
		myImData.hsv_filter.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		//if (pruneseg.empty() == true )
		//	{return;}
	}

void hesapla2(int image_no, int segment_no, int label_no){
		MatND hist;
		labels_array[index] = label_no;


		int ivar = image_no;

		mystream << imagename1 << std::setfill('0') << std::setw(4) << ivar << type;
		string myfilename = mystream.str();
		mystream.str("");
		Mat src = imread(myfilename, CV_LOAD_IMAGE_COLOR);

		Mat dst;
		pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));
		bilateralFilter(dst, myImData.original, 9, 30, 30);

		cvtColor(myImData.original, myImData.intensity, CV_BGR2GRAY);
		cvtColor(myImData.original, myImData.hsvImg, CV_BGR2HSV);

		myImData.h = myImData.original.rows;
		myImData.w = myImData.original.cols;

		vector<Mat> hsvchannels;
		split(myImData.hsvImg, hsvchannels);

		myImData.hsv_filter.push_back(hsvchannels[0]);
		myImData.hsv_filter.push_back(hsvchannels[1]);
		myImData.hsv_filter.push_back(hsvchannels[2]);


		hsvchannels.clear();



		//keep filters in here
		vector<Mat> filters;

		for (int f = 0; f<numberoffilters; f++)
		{
			filters.push_back(readFilter(f, 29));
		}
		for (int i = 0; i<numberoffilters; i++){
			//filter original image and create filtered outputs
			applyFilter(myImData, filters[i], i);

		}
		myImData.filter.push_back(myImData.hsv_filter.at(0)); //Hue filter result added to 5 spatial filter result
		filters.clear();
		//int count[TOTAL_COLOR_Nr];
		//std::fill( count, count+TOTAL_COLOR_Nr, 0);



		for (int k = 0; k < (myImData.h * myImData.w); k++){
			int x = k % (myImData.w);
			int y = (k - x) % (myImData.w - 1);
			bool flag = false;

			for (list<hsvc>::iterator it = col_hash_map.begin(); it != col_hash_map.end(); ++it){


				int val_h = myImData.hsv_filter.at(0).at<uchar>(y, x);
				int val_s = myImData.hsv_filter.at(1).at<uchar>(y, x);
				int val_v = myImData.hsv_filter.at(2).at<uchar>(y, x);


				if (val_h >= it->hlow && val_h <= it->hhigh && val_s >= it->slow &&
					val_s <= it->shigh && val_v >= it->vlow && val_v <= it->vhigh){

					//	c.x = x;
					//	c.y = y;
					//	c.col_info = it->col_name; ++count[(c.col_info-1)];
					//	myImData.hsv_col_info.push_back(c);	

					//	I[y][x]=c.col_info;

					I[y][x] = it->col_name;

					flag = true; break;
				}

			}
		}

		Labeling(labelNr, label, I, Q, EQ, myImData);

		LabelEqualization(EQ, label, myImData, labelColors);


		/*Merge small components with their nearest component*/

		std::unordered_map<int, int> occurrences;

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				++occurrences[label[i][j]];
			}
		}

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				if (occurrences[label[i][j]] < MAX_PxNr_SMALL_AREA) {
					EQ[label[i][j]] = Q[i][j];
					Q[i][j + 1] = Q[i][j];

				}
			}
		}
		occurrences.clear();

		// LabelEqualization(EQ, label, myImData,labelColors);

		vector<int> nIndx;
		int indx = 1;

		while (indx != labelNr + 1){
			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int val = label[i][j];
					if (val == indx){
						contourArray.push_back(Point(j, i));
					}
				}
			}
			if (contourArray.empty() == false){

				myImData.connComp.push_back(contourArray);
				int clrv;

				// int clrv = I[contourArray.at(0).y][contourArray.at(0).x]; //yanlýþ renk birleþtirme olmasýn diye deðiþtiriyorum burayý 07.10.2015
				if (contourArray.size() == 1){
					clrv = I[contourArray.at(0).y][contourArray.at(0).x];
				}
				if (contourArray.size()>1){
					clrv = I[contourArray.at(contourArray.size() - 1).y][contourArray.at(contourArray.size() - 1).x];
				}

				segmentClrs.push_back(clrv);

				if (contourArray.size()> MIN_PxNr_BIG_AREA){
					nIndx.push_back(myImData.connComp.size() - 1);
				}

			}
			++indx;
		}

		for (int nfc = 0; nfc< nIndx.size(); nfc++){

			int numberofcomponents = nIndx.at(nfc);
			Mat component_Img = Mat::zeros(myImData.h, myImData.w, CV_8UC1);
			Mat dilated_component_Img, dst;
			Mat eroded_;
			// Create binary image of big segment
			for (int comp = 0; comp < myImData.connComp.at(numberofcomponents).size(); comp++){
				Point component = myImData.connComp.at(numberofcomponents).at(comp);
				component_Img.at<uchar>(component.y, component.x) = 255;
			}

			dilate(component_Img, dilated_component_Img, dilation_element);

			// Obtain adjacent parts
			cv::bitwise_xor(component_Img, dilated_component_Img, dst);
			//imshow("dst", dst); //adjacent img
			//	waitKey(0);


			vector<Point> nonZeroCoordinates;		//keep adjacent pixels in here
			findNonZero(dst, nonZeroCoordinates);

			int	ColorNr1 = segmentClrs.at(numberofcomponents);
			int ColorNr2;
			int newLabel = label[myImData.connComp.at(numberofcomponents).at(0).y][myImData.connComp.at(numberofcomponents).at(0).x];

			for (int g = 0; g<nonZeroCoordinates.size(); g++){

				Point AdjPoint = nonZeroCoordinates.at(g);
				ColorNr2 = I[AdjPoint.y][AdjPoint.x];
				if (relationLUT[ColorNr1][ColorNr2] == 1){

					EQ[label[AdjPoint.y][AdjPoint.x]] = newLabel;

				}

			}

		}

		LabelEqualization(EQ, label, myImData, labelColors);
		myImData.connComp.clear();

		int nindx = 1;
		while (nindx != labelNr + 1){

			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int nval = label[i][j];
					if (nval == nindx){

						contourArray.push_back(Point(j, i));


					}
				}
			}
			if (contourArray.empty() == false){
				myImData.connComp.push_back(contourArray);
			}
			++nindx;
		}
		
		scalefactor = myImData.connComp.at(segment_no).size();
		features(myImData, segment_no);
		//	cout << mF.hueval << endl << mF.shaperatio << endl;

		//	saveFeatures(featureVector,myImData,ivar);

		Mat goster_filtre(myImData.h, myImData.w, CV_8UC1, Scalar(background));

		vector<vector<float>> svminput;
		vector<float> binvector;


		for (int filter_no = 0; filter_no < numberoffilters; filter_no++)

		{

			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

			{
				Point component = myImData.connComp.at(segment_no).at(d);

				filt_crop.push_back(myImData.filter.at(filter_no).at<uchar>(component.y, component.x));
				//	goster_filtre.at<uchar>(component.y, component.x) = myImData.filter.at(filter_no).at<uchar>(component.y, component.x);


			}

			// Calculate histogram

			calcHist(&filt_crop, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);


			// Show the calculated histogram in command window
			/*double total;
			total = filt_crop.rows * filt_crop.cols;*/
			for (int h = 0; h < histSize; h++)
			{
				float binVal = hist.at<float>(h);
				//cout<<" "<<binVal;
				binvector.push_back(binVal / (scalefactor));
				//	trainingData[index][h] = binVal;

			}
			filt_crop.release();
			hist.release();
		}

			std::unordered_map<int, int> voting;
			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

				{
					Point component = myImData.connComp.at(segment_no).at(d);

					++voting[I[component.y][component.x]];
					
				}

				int currentMax = 0;
				int arg_max = 0;
				for(auto it = voting.cbegin(); it != voting.cend(); ++it ) {
					if (it ->second > currentMax) {
						arg_max = it->first;
						currentMax = it->second;
					}
				}
				//cout << "Value " << arg_max << " occurs " << currentMax << " times " << endl;					
				
			float arg_max_f = arg_max/25;

		binvector.push_back(mF.hueval);
		binvector.push_back(mF.shaperatio);
	//	binvector.push_back(mF.segmentLocation);
		binvector.push_back(arg_max_f);


		for (int s = 0; s < numberofbins; s++){
			trainingData[index][s] = binvector.at(s);
		}

		//482 svm input saved to binvector for an instance

		svminput.push_back(binvector);

		voting.clear();
		nindx = 0;
		nIndx.clear();
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth; pw++){
				label[ph][pw] = 0;
				I[ph][pw] = 0;

			}
		}
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth + 1; pw++){

				Q[ph][pw] = 0;

			}
		}

		for (int eqn = 0; eqn<MAX_EXP_NrOf_LABELS; eqn++){ EQ[eqn] = 0; }


		filt_crop.release();
		binvector.clear();
		segmentClrs.clear();
		contourArray.clear();
		//mychannels.clear();
		//fin_img.release();
		//featureVector.clear();
		//ch0.release();
		//ch1.release();
		//ch2.release();
		mF.AverageFilterOutputs.clear();

		myImData.filter.clear();
		myImData.hsv_filter.clear();
		//	myImData.hsv_col_info.clear();
		myImData.connComp.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		labelNr = 0;
	}

void testyap2(int image_no, int segment_no){
		MatND hist;
		int ivar = image_no;

		mystream << imagename1 << std::setfill('0') << std::setw(4) << ivar << type;
		string myfilename = mystream.str();
		mystream.str("");
		Mat src = imread(myfilename, CV_LOAD_IMAGE_COLOR);

		Mat dst;
		pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));
		bilateralFilter(dst, myImData.original, 9, 30, 30);

		cvtColor(myImData.original, myImData.intensity, CV_BGR2GRAY);
		cvtColor(myImData.original, myImData.hsvImg, CV_BGR2HSV);

		myImData.h = myImData.original.rows;
		myImData.w = myImData.original.cols;

		vector<Mat> hsvchannels;
		split(myImData.hsvImg, hsvchannels);

		myImData.hsv_filter.push_back(hsvchannels[0]);
		myImData.hsv_filter.push_back(hsvchannels[1]);
		myImData.hsv_filter.push_back(hsvchannels[2]);

		hsvchannels.clear();

		//keep filters in here
		vector<Mat> filters;

		for (int f = 0; f<numberoffilters; f++)
		{
			filters.push_back(readFilter(f, 29));
		}
		for (int i = 0; i<numberoffilters; i++){
			//filter original image and create filtered outputs
			applyFilter(myImData, filters[i], i);


		}
		myImData.filter.push_back(myImData.hsv_filter.at(0)); //Hue filter result added to 5 spatial filter result
		filters.clear();

		for (int k = 0; k < (myImData.h * myImData.w); k++){
			int x = k % (myImData.w);
			int y = (k - x) % (myImData.w - 1);
			bool flag = false;

			for (list<hsvc>::iterator it = col_hash_map.begin(); it != col_hash_map.end(); ++it){

				int val_h = myImData.hsv_filter.at(0).at<uchar>(y, x);
				int val_s = myImData.hsv_filter.at(1).at<uchar>(y, x);
				int val_v = myImData.hsv_filter.at(2).at<uchar>(y, x);

				if (val_h >= it->hlow && val_h <= it->hhigh && val_s >= it->slow &&
					val_s <= it->shigh && val_v >= it->vlow && val_v <= it->vhigh){

					I[y][x] = it->col_name;

					flag = true; break;
				}

			}
		}

		Labeling(labelNr, label, I, Q, EQ, myImData);

		LabelEqualization(EQ, label, myImData, labelColors);

		/*Merge small components with their nearest component*/
		std::unordered_map<int, int> occurrences;

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				++occurrences[label[i][j]];
			}
		}

		for (int i = 0; i < myImData.h; ++i){
			for (int j = 0; j < myImData.w; ++j){

				if (occurrences[label[i][j]] < MAX_PxNr_SMALL_AREA) {
					EQ[label[i][j]] = Q[i][j];
					Q[i][j + 1] = Q[i][j];

				}
			}
		}
		occurrences.clear();

		// LabelEqualization(EQ, label, myImData,labelColors);
		vector<int> nIndx;
		int indx = 1;

		while (indx != labelNr + 1){
			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int val = label[i][j];
					if (val == indx){
						contourArray.push_back(Point(j, i));
					}
				}
			}
			if (contourArray.empty() == false){

				myImData.connComp.push_back(contourArray);
				int clrv;

				if (contourArray.size() == 1){
					clrv = I[contourArray.at(0).y][contourArray.at(0).x];
				}
				if (contourArray.size()>1){
					clrv = I[contourArray.at(contourArray.size() - 1).y][contourArray.at(contourArray.size() - 1).x];
				}

				segmentClrs.push_back(clrv);

				if (contourArray.size()> MIN_PxNr_BIG_AREA){
					nIndx.push_back(myImData.connComp.size() - 1);
				}

			}
			++indx;
		}

		for (int nfc = 0; nfc< nIndx.size(); nfc++){

			int numberofcomponents = nIndx.at(nfc);
			Mat component_Img = Mat::zeros(myImData.h, myImData.w, CV_8UC1);
			Mat dilated_component_Img, dst;
			Mat eroded_;
			// Create binary image of big segment
			for (int comp = 0; comp < myImData.connComp.at(numberofcomponents).size(); comp++){
				Point component = myImData.connComp.at(numberofcomponents).at(comp);
				component_Img.at<uchar>(component.y, component.x) = 255;
			}

			dilate(component_Img, dilated_component_Img, dilation_element);

			// Obtain adjacent parts
			cv::bitwise_xor(component_Img, dilated_component_Img, dst);

			vector<Point> nonZeroCoordinates;		//keep adjacent pixels in here
			findNonZero(dst, nonZeroCoordinates);

			int	ColorNr1 = segmentClrs.at(numberofcomponents);
			int ColorNr2;
			int newLabel = label[myImData.connComp.at(numberofcomponents).at(0).y][myImData.connComp.at(numberofcomponents).at(0).x];

			for (int g = 0; g<nonZeroCoordinates.size(); g++){

				Point AdjPoint = nonZeroCoordinates.at(g);
				ColorNr2 = I[AdjPoint.y][AdjPoint.x];
				if (relationLUT[ColorNr1][ColorNr2] == 1){

					EQ[label[AdjPoint.y][AdjPoint.x]] = newLabel;

				}

			}

		}

		LabelEqualization(EQ, label, myImData, labelColors);

		myImData.connComp.clear();

		int nindx = 1;
		while (nindx != labelNr + 1){

			contourArray.clear();
			for (int i = 0; i < myImData.h; i++) {
				for (int j = 0; j < myImData.w; j++) {

					int nval = label[i][j];
					if (nval == nindx){

						contourArray.push_back(Point(j, i));

					}
				}
			}
			if (contourArray.empty() == false){
				myImData.connComp.push_back(contourArray);
			}
			++nindx;
		}

		scalefactor = myImData.connComp.at(segment_no).size();
		features(myImData, segment_no);
		//	cout << mF.hueval << endl << mF.shaperatio << endl;

		// Mat goster_filtre(myImData.h, myImData.w, CV_8UC1, Scalar(background));

		vector<float> binvector;

		for (int filter_no = 0; filter_no < numberoffilters; filter_no++)

		{

			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

			{
				Point component = myImData.connComp.at(segment_no).at(d);

				filt_crop.push_back(myImData.filter.at(filter_no).at<uchar>(component.y, component.x));
				//	goster_filtre.at<uchar>(component.y, component.x) = myImData.filter.at(filter_no).at<uchar>(component.y, component.x);


			}

			// Calculate histogram

			calcHist(&filt_crop, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

			for (int h = 0; h < histSize; h++)
			{
				float binVal = hist.at<float>(h);
				binvector.push_back(binVal / (scalefactor));
			}
			filt_crop.release();
			hist.release();
		}

						std::unordered_map<int, int> voting;
			for (int d = 0; d<myImData.connComp.at(segment_no).size(); d++)

				{
					Point component = myImData.connComp.at(segment_no).at(d);
					++voting[I[component.y][component.x]];
					
				}

				int currentMax = 0;
				int arg_max = 0;
				for(auto it = voting.cbegin(); it != voting.cend(); ++it ) {
					if (it ->second > currentMax) {
						arg_max = it->first;
						currentMax = it->second;
					}
				}
				//cout << "Value " << arg_max << " occurs " << currentMax << " times " << endl;					
				
			float arg_max_f = arg_max/25;


		binvector.push_back(mF.hueval);
		binvector.push_back(mF.shaperatio);
	//	binvector.push_back(mF.segmentLocation);
		binvector.push_back(arg_max_f);

		for (int s = 0; s < numberofbins; s++){
			samples[sindex][s] = binvector.at(s);
		}

		voting.clear();
		nindx = 0;
		nIndx.clear();
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth; pw++){
				label[ph][pw] = 0;
				I[ph][pw] = 0;

			}
		}
		for (int ph = 0; ph<pyrHeight; ph++){
			for (int pw = 0; pw<pyrWidth + 1; pw++){

				Q[ph][pw] = 0;

			}
		}

		for (int eqn = 0; eqn<MAX_EXP_NrOf_LABELS; eqn++){ EQ[eqn] = 0; }

		filt_crop.release();
		binvector.clear();
		segmentClrs.clear();
		contourArray.clear();

		mF.AverageFilterOutputs.clear();

		myImData.filter.clear();
		myImData.hsv_filter.clear();

		myImData.connComp.clear();
		myImData.original.release();
		myImData.intensity.release();
		myImData.hsvImg.release();
		labelNr = 0;

	}

vector<int> segmentsec(vector<vector<Point>> connComp){
	//	MatND hist;
	lops.clear();
	//	int segment_no;
	//	int ivar = image_no;

	std::unordered_map<int, int> voting;

	//	mystream << imagename1 << std::setfill('0') << std::setw(4) << ivar << type;
	//	string myfilename = mystream.str();
	//	mystream.str("");
	//	Mat src = imread(myfilename, CV_LOAD_IMAGE_COLOR);

	//	Mat dst;
	//	pyrDown(src, dst, Size(src.cols / 2, src.rows / 2));
	//	bilateralFilter(dst, myImData.original, 9, 30, 30);

	//	cvtColor(myImData.original, myImData.intensity, CV_BGR2GRAY);
	//	cvtColor(myImData.original, myImData.hsvImg, CV_BGR2HSV);

	//	myImData.h = myImData.original.rows;
	//	myImData.w = myImData.original.cols;

	//	vector<Mat> hsvchannels;
	//	split(myImData.hsvImg, hsvchannels);

	//	myImData.hsv_filter.push_back(hsvchannels[0]);
	//	myImData.hsv_filter.push_back(hsvchannels[1]);
	//	myImData.hsv_filter.push_back(hsvchannels[2]);

	//	hsvchannels.clear();

	//	//keep filters in here
	//	vector<Mat> filters;

	//	for (int f = 0; f<numberoffilters; f++)
	//	{
	//		filters.push_back(readFilter(f, 29));
	//	}
	//	for (int i = 0; i<numberoffilters; i++){
	//		//filter original image and create filtered outputs
	//		applyFilter(myImData, filters[i], i);

	//		
	//	}
	//	myImData.filter.push_back(myImData.hsv_filter.at(0)); //Hue filter result added to 5 spatial filter result
	//	filters.clear();

	//	for (int k = 0; k < (myImData.h * myImData.w); k++){
	//		int x = k % (myImData.w);
	//		int y = (k - x) % (myImData.w - 1);
	//		bool flag = false;

	//		for (list<hsvc>::iterator it = col_hash_map.begin(); it != col_hash_map.end(); ++it){

	//			int val_h = myImData.hsv_filter.at(0).at<uchar>(y, x);
	//			int val_s = myImData.hsv_filter.at(1).at<uchar>(y, x);
	//			int val_v = myImData.hsv_filter.at(2).at<uchar>(y, x);

	//			if (val_h >= it->hlow && val_h <= it->hhigh && val_s >= it->slow &&
	//				val_s <= it->shigh && val_v >= it->vlow && val_v <= it->vhigh){

	//			I[y][x] = it->col_name;

	//				flag = true; break;
	//			}

	//		}
	//	}

	//	Labeling(labelNr, label, I, Q, EQ, myImData);

	//	LabelEqualization(EQ, label, myImData, labelColors);

	//	/*Merge small components with their nearest component*/
	//	std::unordered_map<int, int> occurrences;

	//	for (int i = 0; i < myImData.h; ++i){
	//		for (int j = 0; j < myImData.w; ++j){

	//			++occurrences[label[i][j]];
	//		}
	//	}

	//	for (int i = 0; i < myImData.h; ++i){
	//		for (int j = 0; j < myImData.w; ++j){

	//			if (occurrences[label[i][j]] < MAX_PxNr_SMALL_AREA) {
	//				EQ[label[i][j]] = Q[i][j];
	//				Q[i][j + 1] = Q[i][j];

	//			}
	//		}
	//	}
	//	occurrences.clear();

	//	// LabelEqualization(EQ, label, myImData,labelColors);
	//	vector<int> nIndx;
	//	int indx = 1;

	//	while (indx != labelNr + 1){
	//		contourArray.clear();
	//		for (int i = 0; i < myImData.h; i++) {
	//			for (int j = 0; j < myImData.w; j++) {

	//				int val = label[i][j];
	//				if (val == indx){
	//					contourArray.push_back(Point(j, i));
	//				}
	//			}
	//		}
	//		if (contourArray.empty() == false){

	//			myImData.connComp.push_back(contourArray);
	//			int clrv;

	//			if (contourArray.size() == 1){
	//				clrv = I[contourArray.at(0).y][contourArray.at(0).x];
	//			}
	//			if (contourArray.size()>1){
	//				clrv = I[contourArray.at(contourArray.size() - 1).y][contourArray.at(contourArray.size() - 1).x];
	//			}

	//			segmentClrs.push_back(clrv);

	//			if (contourArray.size()> MIN_PxNr_BIG_AREA){
	//				nIndx.push_back(myImData.connComp.size() - 1);
	//			}

	//		}
	//		++indx;
	//	}

	//	for (int nfc = 0; nfc< nIndx.size(); nfc++){

	//		int numberofcomponents = nIndx.at(nfc);
	//		Mat component_Img = Mat::zeros(myImData.h, myImData.w, CV_8UC1);
	//		Mat dilated_component_Img, dst;
	//		Mat eroded_;
	//		// Create binary image of big segment
	//		for (int comp = 0; comp < myImData.connComp.at(numberofcomponents).size(); comp++){
	//			Point component = myImData.connComp.at(numberofcomponents).at(comp);
	//			component_Img.at<uchar>(component.y, component.x) = 255;
	//		}

	//		dilate(component_Img, dilated_component_Img, dilation_element);

	//		// Obtain adjacent parts
	//		cv::bitwise_xor(component_Img, dilated_component_Img, dst);
	//
	//		vector<Point> nonZeroCoordinates;		//keep adjacent pixels in here
	//		findNonZero(dst, nonZeroCoordinates);

	//		int	ColorNr1 = segmentClrs.at(numberofcomponents);
	//		int ColorNr2;
	//		int newLabel = label[myImData.connComp.at(numberofcomponents).at(0).y][myImData.connComp.at(numberofcomponents).at(0).x];

	//		for (int g = 0; g<nonZeroCoordinates.size(); g++){

	//			Point AdjPoint = nonZeroCoordinates.at(g);
	//			ColorNr2 = I[AdjPoint.y][AdjPoint.x];
	//			if (relationLUT[ColorNr1][ColorNr2] == 1){

	//				EQ[label[AdjPoint.y][AdjPoint.x]] = newLabel;

	//			}

	//		}

	//	}

	//	LabelEqualization(EQ, label, myImData, labelColors);

	//	myImData.connComp.clear();

	//	int nindx = 1;
	//	while (nindx != labelNr + 1){

	//		contourArray.clear();
	//		for (int i = 0; i < myImData.h; i++) {
	//			for (int j = 0; j < myImData.w; j++) {

	//				int nval = label[i][j];
	//				if (nval == nindx){

	//					contourArray.push_back(Point(j, i));

	//				}
	//			}
	//		}
	//		if (contourArray.empty() == false){
	//			myImData.connComp.push_back(contourArray);
	//		}
	//		++nindx;
	//	}


		for (int tSegmentNumber=0; tSegmentNumber < connComp.size(); tSegmentNumber++)
		{
			if (connComp.at(tSegmentNumber).size() > 170){
				
				for (int d = 0; d<connComp.at(tSegmentNumber).size(); d++)

				{
					Point component = connComp.at(tSegmentNumber).at(d);
					++voting[I[component.y][component.x]];
					
				}
					int currentMax = 0;
				int arg_max = 0;
				for(auto it = voting.cbegin(); it != voting.cend(); ++it ) {
					if (it ->second > currentMax) {
						arg_max = it->first;
						currentMax = it->second;
					}
				}

			if(arg_max == 8 || arg_max == 9 || arg_max == 18 || arg_max == 19 || arg_max == 22 || arg_max ==23 || arg_max == 24 || arg_max == 25){
				
					lops.push_back(tSegmentNumber);
			  }

				voting.clear();
			}
		}
		
		//	nindx = 0;
		//nIndx.clear();
		//for (int ph = 0; ph<pyrHeight; ph++){
		//	for (int pw = 0; pw<pyrWidth; pw++){
		//		label[ph][pw] = 0;
		//		I[ph][pw] = 0;

		//	}
		//}
		//for (int ph = 0; ph<pyrHeight; ph++){
		//	for (int pw = 0; pw<pyrWidth + 1; pw++){

		//		Q[ph][pw] = 0;

		//	}
		//}

	//	for (int eqn = 0; eqn<MAX_EXP_NrOf_LABELS; eqn++){ EQ[eqn] = 0; }

	//	filt_crop.release();
	////	binvector.clear();
	//	segmentClrs.clear();
	//	contourArray.clear();

	//	myImData.filter.clear();
	//	myImData.hsv_filter.clear();

	//	myImData.connComp.clear();
	//	myImData.original.release();
	//	myImData.intensity.release();
	//	myImData.hsvImg.release();
	//	labelNr = 0;

		connComp.clear();
			return	lops;

		}
		

	
		

