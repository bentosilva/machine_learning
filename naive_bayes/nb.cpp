/*****************************************
Naive Bayes Method

M.A.P. - Maximize a posterior

RESULT: 1757 correct out of 1779 samples.
RESULT: 1067 class0 samples correct out of 1083 samples
RESULT: 16 class0 samples wrong out of 1083 samples.
RESULT: 690 class1 samples correct out of 696 samples.
RESULT: 6 class1 samples wrong out of 696 samples.
*****************************************/

#include <iostream>
//#include <tr1/random>
#include <time.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstdlib>
#include <math.h>
#include <fstream>

using namespace std;
//using namespace std::tr1;

#define MAX_BIG_BUFF_SIZE 50000 //

// ȫ�ֱ��� - �������ļ��л�ȡ����ʵ��Ϣ 
vector<map<string, int> > wcArrTrain; 	// TRAIN dataset ÿ�����µ� wordcount ����
vector<map<string, int> > wcArrTest;	// TEST  dataset ÿ�����µ� wordcount ����
vector<int> labelsTrain;	// TRAIN dataset ÿ�����µ� label ����
vector<int> labelsTest; 	// TEST  dataset ÿ�����µ� label ����
int count0Train;			// TRAIN dataset label = 0 �������� 
int count1Train;			// TRAIN dataset label = 1 �������� 
int count0Test;				// TEST  dataset label = 0 �������� 
int count1Test;				// TEST  dataset label = 1 �������� 
set<string> vocabulary;		// �����ʿ⣬set������ģ��������ִʿ��˳��Ҳ�ͱ����� wordcount ��ÿ�����ڴʿ��λ�� 

// ȫ�ֱ��� - �� wcArrTrain ���� 
map<string, int> map0;	// word count of label 0 
map<string, int> map1;	// word count of label 1

// ȫ�ֱ��� - ���ָ��� 
double logpr_c0;  				// c0 ��log����
double logpr_c1;  				// c1 ��log����
vector<double> logpr_c0words;	// c0 �����´ʱ���ÿ���ʵĸ��� 
vector<double> logpr_c1words;	// c1 �����´ʱ���ÿ���ʵĸ��� 

 
void run(); 
void evalute_classification();
int load_data_file(char *filename, vector<map<string, int> > *pWCArr, vector<int> *pLabels, int &count0, int &count1); 
void str_explode(const char *pStr, char cSep, std::vector<std::string>& vecItems); 
void init_training_variables(); 
void calculate_prs();
// pseudo count = 1, SMOOTHING skill
int wcmap_2_cntvec(map<string, int> &wcmap, vector<int> *pVec, int pseudo=1);


int main() 
{ 
	run();
    return (0); 
}


void run()
{
	// load ���Լ���ѵ�����ļ������г�ʼ��һ�����ļ�����ȫ�ֱ��� (�ļ�����ǩ��wordcount) 
	int N1 = load_data_file("earn_acp_r8_train.txt", &wcArrTrain, &labelsTrain, count0Train, count1Train);
	int N2 = load_data_file("earn_acp_r8_test.txt", &wcArrTest, &labelsTest, count0Test, count1Test);

	// ������ʵ��ѵ��������ʼ����ǩ�����ȫ�ֱ��� wordcount
	init_training_variables();
	
	// ���� wordcount������ log ���� 
	calculate_prs();
	
	// ���� log ���ʣ��������Լ� 
	evalute_classification();
}


// ���㷨���õ��Ľ���������� 
void evalute_classification ()
{
	int num = labelsTest.size(); 
	int correct = 0;
	int c0to0 = 0, c0to1 = 0, c1to0 = 0, c1to1 = 0;
	int v = vocabulary.size();
	for (int i = 0;  i < num; ++i)
	{
		vector<int> wc;
		// ������ pseudo count������ʱ����Ҫ�� SMOOTHING 
		wcmap_2_cntvec(wcArrTest[i], &wc, 0);
		double pr0 = logpr_c0;
		double pr1 = logpr_c1;
		
		for (int j = 0; j < v; ++j)
		{
			pr0 += wc[j] * logpr_c0words[j];
			pr1 += wc[j] * logpr_c1words[j];
		}
		
		if (pr0 >= pr1)
		{
			if (labelsTest[i] == 0)
			{
				correct ++;
				c0to0 ++;
			}
			else
			{
				c1to0 ++;
			}
		}
		else if (pr0 <= pr1)
		{
			if (labelsTest[i] == 1)
			{
				correct ++;
				c1to1 ++;
			} 
			else
			{
				c0to1 ++;
			}
		}
	}
	cout << "RESULT: " << correct << " correct out of " << num << " samples." << endl;
	cout << "RESULT: " << c0to0 << " class0 samples correct out of " << count0Test << " samples." << endl;
	cout << "RESULT: " << c0to1 << " class0 samples wrong out of " << count0Test << " samples." << endl;
	cout << "RESULT: " << c1to1 << " class1 samples correct out of " << count1Test << " samples." << endl;
	cout << "RESULT: " << c1to0 << " class1 samples wrong out of " << count1Test << " samples." << endl;
}


/*
vector<map<string, int> > *pWCArr : ÿ�����ļ�����Ϣ����ϢΪÿ��wordout    
vector<int> *pLabels              : ÿ��λ�ñ���һ���ļ�����Ϣ����ϢΪ���ļ���Ӧ�� label (0/1) 
int &count0, int &count1          : �����ļ��У�һ�����ٸ� label = 0 ���ļ������ٸ� label = 1 ���ļ� 

���ļ���˳���ʼ���˴ʿ� vocabulary 
*/
int load_data_file(char *filename, vector<map<string, int> > *pWCArr, vector<int> *pLabels, int &count0, int &count1)
{	
	count0 = 0;
	count1 = 0;
	pLabels->clear();
	pWCArr->clear();
	 
	char cTmpBuf[MAX_BIG_BUFF_SIZE] = {0};
	std::ifstream ifsm;
	ifsm.open(filename, ios_base::in);
	if (!ifsm)
	{
		cerr << "ERROR: cannot open file " << filename << endl;
		exit(111);
	}

	while( ifsm.getline(cTmpBuf, MAX_BIG_BUFF_SIZE, '\n') )
	{
		vector<string> vec1;
		str_explode(cTmpBuf, '\t', vec1);
		if (vec1.size() == 2 && vec1[1] != "")
		{
			// update labels
			if (vec1[0] == "earn")
			{
				pLabels->push_back(0);
				count0 ++;
			}
			else
			{
				pLabels->push_back(1);
				count1 ++;
			}
			
			// update word account
			map<string, int> wc;
			wc.clear();
			vector<string> vec2;
			str_explode(vec1[1].c_str(), ' ', vec2);
			int vec2len = vec2.size();
			for (int i = 0; i < vec2len; ++i)
			{
				// ����ʵ� 
				vocabulary.insert(vec2[i]);
				// ���� word count map  
				map<string, int>::iterator it = wc.find(vec2[i]);
				if (it == wc.end())
				{
					wc.insert(map<string, int>::value_type(vec2[i], 1));
				} 
				else
				{
					it->second ++;
				}
			}
			pWCArr->push_back(wc); 
		}
	}

	ifsm.close();
	ifsm.clear();	
	cout << "File " << filename << " -  earn count: " << count0 << "    acq count: " << count1 << endl;	
	return count0 + count1;
}


// explode 
void str_explode(const char *pStr, char cSep, std::vector<std::string>& vecItems)
{
     std::string strTmpcheck(pStr);
     string::size_type pos; 
     std::string strHead;
     
     vecItems.clear();
     while (1)
     {
         pos = strTmpcheck.find(cSep);
         if (pos != strTmpcheck.npos)
         {
              strHead = strTmpcheck.substr(0, pos);
              vecItems.push_back(strHead);
              strTmpcheck = strTmpcheck.substr(pos + 1);
         }
         else 
         {
              vecItems.push_back(strTmpcheck);
              break;
         } 
     }
     return;
} 


/*
����training�ļ����õ� map0, map1
*/ 
void init_training_variables()
{
	// ѵ���������� 
	int nTrain = labelsTrain.size();
	map0.clear();
	map1.clear();
	
	// ����ѵ�������� 
	for (int i = 0; i < nTrain; ++i)
	{
		if (labelsTrain[i] == 0)
		{
			// ���������µ� word count 
			for (map<string, int>::iterator it = wcArrTrain[i].begin(); it != wcArrTrain[i].end(); ++it) 
			{
				// �� map0 �в��� word (it->first) 
				map<string, int>::iterator iter = map0.find(it->first); 
				// û�ҵ���ֱ�Ӳ��� 
				if (iter == map0.end())
				{
					map0.insert(map<string, int>::value_type(it->first, it->second)); 
				}
				// �ҵ��ˣ����� count 
				else
				{
					iter->second += it->second; 
				}
			}
		}
		else
		{
			for (map<string, int>::iterator it = wcArrTrain[i].begin(); it != wcArrTrain[i].end(); ++it) 
			{
				map<string, int>::iterator iter = map1.find(it->first); 
				if (iter == map1.end())
				{
					map1.insert(map<string, int>::value_type(it->first, it->second)); 
				}
				else
				{
					iter->second += it->second; 
				}
			}
		}
	}
}


// ���� logpr_c0��logpr_c1��logpr_c0words��logpr_c1words 
void calculate_prs()
{
	// (double) �ǳ���Ҫ������ͱ�� 0/1 �� 
	double pr0 = (double)count0Train / (count0Train + count1Train);
	logpr_c0 = log(pr0);
	logpr_c1 = log(1 - pr0);

	vector<int> vec0, vec1;
	int sum0 = wcmap_2_cntvec(map0, &vec0);		
	int sum1 = wcmap_2_cntvec(map1, &vec1);	
	
	int v = vocabulary.size();
	for (int i = 0; i < v; ++i)
	{
		logpr_c0words.push_back(log((double)vec0[i] / sum0)); 
		logpr_c1words.push_back(log((double)vec1[i] / sum1)); 
	}	
}


// �� wordcount map ת��Ϊ count vector��word ˳������ vocabulary �е�˳�򣬹��� V ��countԪ�أ����ﲻ��Ҫָ������ȱʡֵ 
int wcmap_2_cntvec(map<string, int> &wcmap, vector<int> *pVec, int pseudo)
{
	pVec->clear();
	int sum = 0;
	// set is sorted by design :)
	for(set<string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++)
	{
		map<string, int>::iterator iter = wcmap.find(*it);
		int count = 0;
		if (iter == wcmap.end())
		{
			count = pseudo;
		}
		else
		{
			count = iter->second + pseudo;
		}
		pVec->push_back(count);
		sum += count;
	} 
	return sum;
} 


