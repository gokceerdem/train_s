#ifndef getdataclass_H
#define getdataclass_H

class getdataclass 
{
public:
	float labelsArr[250];
	float trainArr[250][2];
	float testArr[100][2];

	void getTrainData(String trndt){
		int cidx = 0;
		string line;
		ifstream file(trndt);

		while (getline(file, line))
		{
			stringstream   linestream(line);
			string        data;
			int imNO, sNo, lNO;

			getline(linestream, data, '\t');
			linestream >> sNo >> lNO;

			/* Array implementation*/
			trainArr[cidx][0] = atoi(line.c_str());
			trainArr[cidx][1] = sNo;
			labelsArr[cidx] = lNO;

			++cidx;

		}
		file.close();

	}
	void getTestData(String tstdt){
		int cidx = 0;
		string line;
		ifstream file(tstdt);

		while (getline(file, line))
		{
			stringstream   linestream(line);
			string        data;
			int imNO, sNo;

			getline(linestream, data, '\t');
			linestream >> sNo ;

			/* Array implementation*/
			testArr[cidx][0] = atoi(line.c_str());
			testArr[cidx][1] = sNo;
			++cidx;

		}
		file.close();

	}



};
#endif