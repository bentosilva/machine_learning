/************************************************************************
Gibbs Naive Bayes

Instead of use MAP probability, it will sample posterior probabilities  
p(c|doc) = p(c) * p(doc|c) / p(doc)
 - sample Lables with Gibbs MCMC 
 - sample thetas with Dirichlet dist.
Use Expected value with those samples

File earn_acp_r8_train.txt -  earn count: 2840    acq count: 1596
File earn_acp_r8_test.txt -  earn count: 1083    acq count: 696
size of Vocabulary: 14308

-----
Iteration 10 | burn in 3 | lap 2 Result:   (worse than Naive Bayes)
	
RESULT: 1746 correct out of 1779 samples.
RESULT: 1056 class0 samples correct out of 1083 samples.
RESULT: 27 class0 samples wrong out of 1083 samples.
RESULT: 690 class1 samples correct out of 696 samples.
RESULT: 6 class1 samples wrong out of 696 samples.

-----
Iteration 50 | burn in 15 | lap 5 Result:   (still worse than Naive Bayes)
	
RESULT: 1752 correct out of 1779 samples.
RESULT: 1062 class0 samples correct out of 1083 samples.
RESULT: 21 class0 samples wrong out of 1083 samples.
RESULT: 690 class1 samples correct out of 696 samples.
RESULT: 6 class1 samples wrong out of 696 samples.

************************************************************************/
#include <iostream>
#include <tr1/random>
#include <time.h>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cstdlib>
#include <math.h>
#include <fstream>

using namespace std;
using namespace std::tr1;

#define MAX_ITERATION 50
#define BURN_IN 15
#define LAG 5 
#define PSEUDO_COUNT 1	// pseudo for theta
#define PSEUDO_PI0   1	// pseudo for pi 0
#define PSEUDO_PI1   1	// pseudo for pi 1
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

// ȫ�ֱ��� - �����������б����㱻���� 
vector<int> labelsTestPredict;  	// TEST dataset ÿ�����µ� "Ԥ��" label ����
int c0; 	// count of label 0
int c1; 	// count of label 1
map<string, int> map0;	// word count of label 0 
map<string, int> map1;	// word count of label 1
vector<double> theta0;				
vector<double> theta1;

// ȫ�ֱ��� - �ռ�������������ڼ����㷨
vector<int> labelsTestCollection; 
int cntCollection;

 
void run(); 
void evalute_classification();
int load_data_file(char *filename, vector<map<string, int> > *pWCArr, vector<int> *pLabels, int &count0, int &count1); 
void str_explode(const char *pStr, char cSep, std::vector<std::string>& vecItems); 
void init_training_variables(); 
void sample_gamma(vector<int> &pars, int ncnt, int seed, vector<double> *pResult); 
void removeDoc(int i);
void addDoc(int i);
void wcmap_2_cntvec(map<string, int> &wcmap, vector<int> *pVec, int pseudo=PSEUDO_COUNT);
double calculate_pr(int cx, map<string, int> &wcmap, int n, int v, int spi_x, int spi_other, vector<double> &theta); 
bool sample_uniform(double par, int seed); 


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
	// �������� 
	int N = N1 + N2;
	
	// ��ʼ�� ���Լ��� "Ԥ��" labels ����
	// ���Լ�����ʵ labels ��������ֻ��������У�鹤�� 
	srand(time(0));
	labelsTestPredict.clear();
	for (int i = 0; i < N2; ++i)
	{
		labelsTestPredict.push_back(rand()%2);
	} 
	
	// ������ʵ��ѵ������ǩ �� Ԥ�ƵĲ��Լ���ǩ ����ʼ����ǩ�����ȫ�ֱ��� (��ǩ����������wordcount)
	init_training_variables();
	
	// ��ʼ�� theta0 & theta1 , pseudo count = 1.0;
	int V = vocabulary.size();
	// V = 14308 in this case
	cout << "size of Vocabulary: " << V << endl;
	vector<int> pars;
	for (int i = 0; i < V; ++i)
	{
		pars.push_back(PSEUDO_COUNT);
	} 
	sample_gamma(pars, V, 1, &theta0);
	sample_gamma(pars, V, 2, &theta1);
	 
	// ��ʼ���������	
	cntCollection = 0;
	labelsTestCollection.clear();
	for (int j = 0; j < N2; ++j) 
	{
		labelsTestCollection.push_back(0); 
	}
	
	// ��ʼ����ϣ���ʼ���� MAX_ITERATION ������ 
	for (int i = 0; i < MAX_ITERATION; ++i)
	{
		cout << "Iteration " << i+1 << " is ongoing ..." << endl;
		// ֻ�Բ��Լ������±�ǩ 
		for (int j = 0; j < N2; ++j)
		{
			// ����ȥ���� j �����¶Ա�ǩ��������wordcount��Ӱ�� 
			removeDoc(j);
			
			// Ȼ������� j ������Ӧ�������ĸ���ǩ 
			// assign  labelsTestPredict[j] = xxx
			double logp0 = calculate_pr(c0, wcArrTest[j], N, V, PSEUDO_PI0, PSEUDO_PI1, theta0);
			double logp1 = calculate_pr(c1, wcArrTest[j], N, V, PSEUDO_PI1, PSEUDO_PI0, theta1);
			// ��ʱ�򣬸��ʲ��ǳ�֮��  log(999999) ~= 13.815509557963773
			// ��ô�Ͳ���sample�� 
			if (logp0 - logp1 > 13.81)
			{
				labelsTestPredict[j] = 0; 
			}  
			else if (logp1 - logp0 > 13.81)
			{
				labelsTestPredict[j] = 1; 
			}
			// ���û��ô��sample it
			// log(p0/p1) = log(p0) - log(p1) ==> p0/p1 = exp(logp0 - logp1)
			// ==> p0/p0+p1 = 1 / ( 1 + exp(logp1 - logp0) )
			double normalized_p0 = 1. / (1. + exp(logp1 - logp0));
			labelsTestPredict[j] = sample_uniform(normalized_p0, j) ? 0 : 1;
			
			// �������µı�ǩ������� j �����¶Ա�ǩ��������wordcount��Ӱ��
			addDoc(j);
		}
		
		// �Ӹ��µ� map0 �л�ȡÿ���ʵ�count+PSEUDO_COUNT 
		vector<int> c0vec;
		wcmap_2_cntvec(map0, &c0vec);
		
		// ���� theta0
		sample_gamma(c0vec, V, 3+i, &theta0);
		c0vec.clear(); 
		
		// �Ӹ��µ� map1 �л�ȡÿ���ʵ�count+PSEUDO_COUNT 
		vector<int> c1vec;
		wcmap_2_cntvec(map1, &c1vec);
		
		// ���� theta1
		sample_gamma(c1vec, V, 4+i, &theta1);
		c1vec.clear();
		
		
		// һ�� iteration ��ɣ�����Ƿ��¼���εĽ��
		if (i >= BURN_IN && i % LAG == 0)
		{
			cntCollection ++;
			for (int j = 0; j < N2; ++j) 
			{
				labelsTestCollection[j] += labelsTestPredict[j]; 
			}
		} 
	}
	
	evalute_classification();
}


// ���㷨���õ��Ľ���������� 
void evalute_classification ()
{
	int num = labelsTestCollection.size(); 
	int correct = 0;
	int c0to0 = 0, c1to1 = 0, c0to1 = 0, c1to0 = 0;
	for (int i = 0;  i < num; ++i)
	{
		if ((double)labelsTestCollection[i] / cntCollection >= 0.5)
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
		else if ((double)labelsTestCollection[i] / cntCollection < 0.5)
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
��ʼ�� c0, c1, map0, map1
�ⲽ������ load_data_file ��������������Ч�ʣ������� load_data_file �ٱ���һ��õ��İ����µ� wordcount ���� pWCArr��Ч�ʵ�һЩ 
���ǣ��������ڣ����� TEST dataset���޷��� load_data_file ������������Ϊ��ʱ�� labels ����ʵ�� labels 
Ȼ����� TEST dataset��������Ҫ�������ǳ�ʼ���� labels ���õ������յ�ÿ�� labels �µ� word count  

����Ҫ���������ֱ��ʹ�� load_data_file ������ʼ������Щȫ�ֱ������Լ������ʼ���� labelsTestPredict �������� 
*/ 
void init_training_variables()
{
	// ѵ���������� N1
	int nTrain = labelsTrain.size();
	// ���Լ������� N2
	int nTest  = labelsTestPredict.size(); 
	// c0 �� c1 ֱ�ӿ���ʹ��ѵ������ count0Train & count1Train ȫ�ֱ�����ʼ�� 
	c0 = count0Train;
	c1 = count1Train;
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
	
	// �������Լ����� 
	for (int i = 0; i < nTest; ++i)
	{
		if (labelsTestPredict[i] == 0)
		{
			for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
			{
				map<string, int>::iterator iter = map0.find(it->first); 
				if (iter == map0.end())
				{
					map0.insert(map<string, int>::value_type(it->first, it->second)); 
				}
				else
				{
					iter->second += it->second; 
				}
			}
			// ���� c0
			c0 ++; 
		}
		else
		{
			for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
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
			// ���� c1
			c1 ++; 
		}
	}
}


/* 
pars: ������һ��gamma�ֲ��Ĳ�����ÿ���������ݲ�����Ӧ��gamma�ֲ���ȡ�������Ǵ�ͬһ���ֲ�ȡ 
ncnt: ��Ҫ�����ĸ��� 
gamma�ֲ���scaling_parameter �̶�Ϊ 1, �ʴ�ֻ��Ҫһ������Ϳ����� 
seed: �������ò�ͬ��������� 
*/
void sample_gamma(vector<int> &pars, int ncnt, int seed, vector<double> *pResult)
{
	time_t now;
	time(&now);
    std::tr1::ranlux64_base_01 eng; 
    eng.seed(now + seed);
    
    double sum = 0.0;
    vector<double> ys;
    // sample phase
    for (int i = 0; i < ncnt; ++i)
    {
	    std::tr1::gamma_distribution<double> dist(pars[i]); 
	    dist.reset(); // discard any cached values 
    	double sample = dist(eng);
    	ys.push_back(sample);
    	sum += sample;
    }
    
    // normalization phase
    pResult->clear();
    for (int i = 0; i < ncnt; ++i)
    {
    	pResult->push_back(ys[i] / sum);
    }
}


// ��ȫ��(c0,c1,map0,map1)��ȥ���� i �����Լ����µ�Ӱ��
void removeDoc(int i)
{
	// label��Ӧ����������Ҫ��1, ҲҪ�Ѷ�Ӧ���µ� wordcountȥ�� 
	if (labelsTestPredict[i] == 0)
	{
		c0--;
		for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
		{
			map<string, int>::iterator iter = map0.find(it->first); 
			if (iter == map0.end())
			{
				cout << "Word: " << it->first << " should be in map0";
				exit(110); 
			}
			else
			{
				iter->second -= it->second; 
			}
		}
	} 
	else
	{
		c1--;
		for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
		{
			map<string, int>::iterator iter = map1.find(it->first); 
			if (iter == map1.end())
			{
				cout << "Word: " << it->first << " should be in map1";
				exit(110); 
			}
			else
			{
				iter->second -= it->second; 
			}
		}
	}
}


// ��ȫ��(c0,c1,map0,map1)�м��ϵ� i �����Լ����µ�Ӱ��
void addDoc(int i)
{
	// label��Ӧ����������Ҫ+1, ҲҪ�Ѷ�Ӧ���µ� wordcount ���� 
	if (labelsTestPredict[i] == 0)
	{
		c0++;
		for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
		{
			map<string, int>::iterator iter = map0.find(it->first); 
			if (iter == map0.end())
			{
				map0.insert(map<string, int>::value_type(it->first, it->second)); 
			}
			else
			{
				iter->second += it->second; 
			}
		}
	} 
	else
	{
		c1++;
		for (map<string, int>::iterator it = wcArrTest[i].begin(); it != wcArrTest[i].end(); ++it) 
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


// �� wordcount map ת��Ϊ count vector��word ˳������ vocabulary �е�˳�򣬹��� V ��countԪ�أ����ﲻ��Ҫָ������ȱʡֵ 
void wcmap_2_cntvec(map<string, int> &wcmap, vector<int> *pVec, int pseudo)
{
	pVec->clear();
	
	for(set<string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++)
	{
		map<string, int>::iterator iter = wcmap.find(*it);
		// �ôʲ��� map �� 
		if (iter == wcmap.end())
		{
			pVec->push_back(0 + pseudo);
		}
		else
		{
			pVec->push_back(iter->second + pseudo);
		}
	} 
} 


// calculate the probability of Lable j 
// use log instead, in case the double overflow (with too many multiply operations)
double calculate_pr(int cx, map<string, int> &wcmap, int n, int v, int spi_x, int spi_other, vector<double> &theta)
{
	double res = log(cx + spi_x - 1) - log(n + spi_x + spi_other -1);
	
	vector<int> cvec;
	// don't use pseudo count here
	wcmap_2_cntvec(wcmap, &cvec, 0);
	for (int i = 0; i < v; ++i)
	{
		res += cvec[i] * log(theta[i]);
	}
	
	// log of the probability
	return res;
}


// ��������� par �����ʣ�Ȼ���������� < par, ��ô���� true�����򷵻� false 
bool sample_uniform(double par, int seed)
{
	time_t now;
	time(&now);
    std::tr1::ranlux64_base_01 eng; 
    eng.seed(now + seed);
    
    std::tr1::uniform_real<double> dist(0, 1);
	dist.reset(); // discard any cached values 
    double sample = dist(eng);
    
    if (sample <= par)
    {
    	return true;
    }
    else
    {
    	return false;
    }
}

