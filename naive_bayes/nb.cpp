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

// 全局变量 - 从数据文件中获取的真实信息 
vector<map<string, int> > wcArrTrain; 	// TRAIN dataset 每个文章的 wordcount 集合
vector<map<string, int> > wcArrTest;	// TEST  dataset 每个文章的 wordcount 集合
vector<int> labelsTrain;	// TRAIN dataset 每个文章的 label 集合
vector<int> labelsTest; 	// TEST  dataset 每个文章的 label 集合
int count0Train;			// TRAIN dataset label = 0 的文章数 
int count1Train;			// TRAIN dataset label = 1 的文章数 
int count0Test;				// TEST  dataset label = 0 的文章数 
int count1Test;				// TEST  dataset label = 1 的文章数 
set<string> vocabulary;		// 整个词库，set是排序的，这样保持词库的顺序，也就保持了 wordcount 中每个词在词库的位置 

// 全局变量 - 由 wcArrTrain 汇总 
map<string, int> map0;	// word count of label 0 
map<string, int> map1;	// word count of label 1

// 全局变量 - 各种概率 
double logpr_c0;  				// c0 的log概率
double logpr_c1;  				// c1 的log概率
vector<double> logpr_c0words;	// c0 分类下词表中每个词的概率 
vector<double> logpr_c1words;	// c1 分类下词表中每个词的概率 

 
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
	// load 测试集和训练集文件，从中初始化一部分文件级的全局变量 (文件级标签和wordcount) 
	int N1 = load_data_file("earn_acp_r8_train.txt", &wcArrTrain, &labelsTrain, count0Train, count1Train);
	int N2 = load_data_file("earn_acp_r8_test.txt", &wcArrTest, &labelsTest, count0Test, count1Test);

	// 根据真实的训练集来初始化标签级别的全局变量 wordcount
	init_training_variables();
	
	// 根据 wordcount，计算 log 概率 
	calculate_prs();
	
	// 根据 log 概率，评估测试集 
	evalute_classification();
}


// 对算法所得到的结果进行评估 
void evalute_classification ()
{
	int num = labelsTest.size(); 
	int correct = 0;
	int c0to0 = 0, c0to1 = 0, c1to0 = 0, c1to1 = 0;
	int v = vocabulary.size();
	for (int i = 0;  i < num; ++i)
	{
		vector<int> wc;
		// 不加入 pseudo count，评估时不需要做 SMOOTHING 
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
vector<map<string, int> > *pWCArr : 每个个文件的信息，信息为每个wordout    
vector<int> *pLabels              : 每个位置保存一个文件的信息，信息为该文件对应的 label (0/1) 
int &count0, int &count1          : 数据文件中，一共多少个 label = 0 的文件，多少个 label = 1 的文件 

本文件还顺便初始化了词库 vocabulary 
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
				// 加入词典 
				vocabulary.insert(vec2[i]);
				// 加入 word count map  
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
汇总training文件，得到 map0, map1
*/ 
void init_training_variables()
{
	// 训练集文章数 
	int nTrain = labelsTrain.size();
	map0.clear();
	map1.clear();
	
	// 遍历训练集文章 
	for (int i = 0; i < nTrain; ++i)
	{
		if (labelsTrain[i] == 0)
		{
			// 遍历该文章的 word count 
			for (map<string, int>::iterator it = wcArrTrain[i].begin(); it != wcArrTrain[i].end(); ++it) 
			{
				// 在 map0 中查找 word (it->first) 
				map<string, int>::iterator iter = map0.find(it->first); 
				// 没找到，直接插入 
				if (iter == map0.end())
				{
					map0.insert(map<string, int>::value_type(it->first, it->second)); 
				}
				// 找到了，更新 count 
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


// 计算 logpr_c0、logpr_c1、logpr_c0words、logpr_c1words 
void calculate_prs()
{
	// (double) 非常重要，否则就变成 0/1 了 
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


// 把 wordcount map 转换为 count vector，word 顺序依据 vocabulary 中的顺序，共计 V 个count元素，这里不需要指定参数缺省值 
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


