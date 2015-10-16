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

// 全局变量 - 在整个流程中被计算被处理 
vector<int> labelsTestPredict;  	// TEST dataset 每个文章的 "预计" label 集合
int c0; 	// count of label 0
int c1; 	// count of label 1
map<string, int> map0;	// word count of label 0 
map<string, int> map1;	// word count of label 1
vector<double> theta0;				
vector<double> theta1;

// 全局变量 - 收集整个结果，用于检验算法
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
	// load 测试集和训练集文件，从中初始化一部分文件级的全局变量 (文件级标签和wordcount) 
	int N1 = load_data_file("earn_acp_r8_train.txt", &wcArrTrain, &labelsTrain, count0Train, count1Train);
	int N2 = load_data_file("earn_acp_r8_test.txt", &wcArrTest, &labelsTest, count0Test, count1Test);
	// 文章总数 
	int N = N1 + N2;
	
	// 初始化 测试集的 "预计" labels 集合
	// 测试集的真实 labels 集合数据只用于最后的校验工作 
	srand(time(0));
	labelsTestPredict.clear();
	for (int i = 0; i < N2; ++i)
	{
		labelsTestPredict.push_back(rand()%2);
	} 
	
	// 根据真实的训练集标签 和 预计的测试集标签 来初始化标签级别的全局变量 (标签级文章数和wordcount)
	init_training_variables();
	
	// 初始化 theta0 & theta1 , pseudo count = 1.0;
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
	 
	// 初始化结果变量	
	cntCollection = 0;
	labelsTestCollection.clear();
	for (int j = 0; j < N2; ++j) 
	{
		labelsTestCollection.push_back(0); 
	}
	
	// 初始化完毕，开始运行 MAX_ITERATION 个迭代 
	for (int i = 0; i < MAX_ITERATION; ++i)
	{
		cout << "Iteration " << i+1 << " is ongoing ..." << endl;
		// 只对测试集来更新标签 
		for (int j = 0; j < N2; ++j)
		{
			// 首先去掉第 j 个文章对标签文章数和wordcount的影响 
			removeDoc(j);
			
			// 然后决定第 j 个文章应该属于哪个标签 
			// assign  labelsTestPredict[j] = xxx
			double logp0 = calculate_pr(c0, wcArrTest[j], N, V, PSEUDO_PI0, PSEUDO_PI1, theta0);
			double logp1 = calculate_pr(c1, wcArrTest[j], N, V, PSEUDO_PI1, PSEUDO_PI0, theta1);
			// 有时候，概率差距非常之大  log(999999) ~= 13.815509557963773
			// 那么就不用sample了 
			if (logp0 - logp1 > 13.81)
			{
				labelsTestPredict[j] = 0; 
			}  
			else if (logp1 - logp0 > 13.81)
			{
				labelsTestPredict[j] = 1; 
			}
			// 差距没那么大，sample it
			// log(p0/p1) = log(p0) - log(p1) ==> p0/p1 = exp(logp0 - logp1)
			// ==> p0/p0+p1 = 1 / ( 1 + exp(logp1 - logp0) )
			double normalized_p0 = 1. / (1. + exp(logp1 - logp0));
			labelsTestPredict[j] = sample_uniform(normalized_p0, j) ? 0 : 1;
			
			// 最后根据新的标签，加入第 j 个文章对标签文章数和wordcount的影响
			addDoc(j);
		}
		
		// 从更新的 map0 中获取每个词的count+PSEUDO_COUNT 
		vector<int> c0vec;
		wcmap_2_cntvec(map0, &c0vec);
		
		// 更新 theta0
		sample_gamma(c0vec, V, 3+i, &theta0);
		c0vec.clear(); 
		
		// 从更新的 map1 中获取每个词的count+PSEUDO_COUNT 
		vector<int> c1vec;
		wcmap_2_cntvec(map1, &c1vec);
		
		// 更新 theta1
		sample_gamma(c1vec, V, 4+i, &theta1);
		c1vec.clear();
		
		
		// 一次 iteration 完成，检查是否记录本次的结果
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


// 对算法所得到的结果进行评估 
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
初始化 c0, c1, map0, map1
这步本来在 load_data_file 函数中来做最有效率；调用完 load_data_file 再遍历一遍得到的按文章的 wordcount 集合 pWCArr，效率低一些 
但是，问题在于，对于 TEST dataset，无法在 load_data_file 函数中做，因为此时的 labels 是真实的 labels 
然后对于 TEST dataset，我们需要按照我们初始化的 labels 来得到的最终的每个 labels 下的 word count  

不需要输入参数，直接使用 load_data_file 函数初始化的那些全局变量，以及随机初始化的 labelsTestPredict 变量即可 
*/ 
void init_training_variables()
{
	// 训练集文章数 N1
	int nTrain = labelsTrain.size();
	// 测试集文章数 N2
	int nTest  = labelsTestPredict.size(); 
	// c0 和 c1 直接可以使用训练集的 count0Train & count1Train 全局变量初始化 
	c0 = count0Train;
	c1 = count1Train;
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
	
	// 遍历测试集文章 
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
			// 更新 c0
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
			// 更新 c1
			c1 ++; 
		}
	}
}


/* 
pars: 给定的一组gamma分布的参数，每个采样根据参数对应的gamma分布来取，而不是从同一个分布取 
ncnt: 需要采样的个数 
gamma分布的scaling_parameter 固定为 1, 故此只需要一组参数就可以了 
seed: 用来设置不同的随机种子 
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


// 从全集(c0,c1,map0,map1)中去掉第 i 个测试集文章的影响
void removeDoc(int i)
{
	// label对应的文章数需要减1, 也要把对应文章的 wordcount去掉 
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


// 在全集(c0,c1,map0,map1)中加上第 i 个测试集文章的影响
void addDoc(int i)
{
	// label对应的文章数需要+1, 也要把对应文章的 wordcount 加上 
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


// 把 wordcount map 转换为 count vector，word 顺序依据 vocabulary 中的顺序，共计 V 个count元素，这里不需要指定参数缺省值 
void wcmap_2_cntvec(map<string, int> &wcmap, vector<int> *pVec, int pseudo)
{
	pVec->clear();
	
	for(set<string>::iterator it = vocabulary.begin(); it != vocabulary.end(); it++)
	{
		map<string, int>::iterator iter = wcmap.find(*it);
		// 该词不在 map 中 
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


// 按输入参数 par 做概率，然后采样；如果 < par, 那么返回 true；否则返回 false 
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

