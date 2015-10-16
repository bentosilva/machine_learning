/************************************************************************
Gibbs LDA

File earn_acp_r8_train.txt -  earn count: 2840    acq count: 1596
File earn_acp_r8_test.txt -  earn count: 1083    acq count: 696
size of Vocabulary: 14308

这个版本里面采用 Unsupervised 学习，这样训练集和测试集不区分，我们同等看待来处理
这样，整个代码和文章中的逻辑完全一致 
最后得到的结果和已知结果比较

由于采用了更简单的数据结构，故此运行比 Gibbs Naive Bayes 更快 
 
-----
Iteration 10 | burn in 3 | lap 2 Result:   (Pretty Bad)
	
RESULT: 3786 correct out of 6215 samples.  (取这个)
    OR: 2429 correct out of 6215 samples.
    
-----
Iteration 50 | burn in 15 | lap 5 Result:   (Better)
	
RESULT: 1980 correct out of 6215 samples.
    OR: 4235 correct out of 6215 samples.  (取这个)
    
-----
Iteration 500 | burn in 100 | lap 8 | ALPHA 25 | BETA 0.01  Result:   (Much Better)

RESULT: 5285 correct out of 6215 samples.  (取这个) 
    OR: 930 correct out of 6215 samples.
    
-----
Iteration 500 | burn in 100 | lap 8 | ALPHA 25 | BETA 1.  Result:   (not change much)
RESULT: 5381 correct out of 6215 samples.  (取这个) 
    OR: 834 correct out of 6215 samples.    

直到此处，所有的结果都没有 gibbs_naivebayes 的结果好
RESULT: 1752 correct out of 1779 samples.  


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

#define MAX_ITERATION 500
#define BURN_IN 100
#define LAG 10 
#define ALPHA 25	// alpha  50/K = 50/2 = 25，两个分类 
#define BETA  0.01  // beta 0.01
#define MAX_BIG_BUFF_SIZE 50000 //

// 全局变量 - 从数据文件中获取的真实信息 
vector<vector<string> > wcArrTrain; 	// TRAIN dataset 每个文章的 word 集合
vector<vector<string> > wcArrTest;		// TEST  dataset 每个文章的 word 集合
vector<int> labelsTrain;	// TRAIN dataset 每个文章的 label 集合
vector<int> labelsTest; 	// TEST  dataset 每个文章的 label 集合

set<string> vocabulary;		// 整个词库，set是排序的，这样保持词库的顺序，也就保持了 wordcount 中每个词在词库的位置 
map<string, int> vocLoc;	// 整个词库，每个词和他的位置 

// 全局变量 - 在整个流程中被计算被处理 
vector<vector<int> > wlArrTrain; 		// TRAIN dataset 每个文章的 label集合 (每个词的label)
vector<vector<int> > wlArrTest;			// TEST  dataset 每个文章的 label集合 (每个词的label) 
vector<int> nm0;	// 每个元素是对应文章采样为 topic 0 的词总数，维度为文章总个数 
vector<int> nm1;	// 每个元素是对应文章采样为 topic 1 的词总数 
vector<int> nmttl;	// 每个元素是对应文章的词总数，维度为文章总个数;   nm0 / nmttl = doc m 为 topic 0 的几率 (相对于topic 1)
vector<int> nk0;	// 每个元素是对应词采样为 topic 0 的总次数，维度为词库中词个数 
vector<int> nk1;	// 每个元素是对应词采样为 topic 1 的总次数 
int nkttl0;			// 所有词中采样为 topic 0 的总次数 	  nk0 / nkttl0 为 词 k 在 topic 0 下的几率 (相对于其他词) 
int nkttl1; 		// 所有词中采样为 topic 1 的总次数;   

// 全局变量 - 收集整个结果，用于检验算法
vector<int> labelsTrainCollection; 
vector<int> labelsTestCollection; 
int cntCollection;

 
void run(); 
int load_data_file(char *filename, vector<vector<string> > *pWCArr, vector<int> *pLabels); 
void str_explode(const char *pStr, char cSep, std::vector<std::string>& vecItems); 
void init_corpus();
void remove_word(int label, int doc_idx, int word_idx);
void add_word(int label, int doc_idx, int word_idx);
double calculate_logpr(int label, int doc_idx, int word_idx, int vocLen);
void init_training_variables(); 
int get_word_idx(string s);
bool sample_uniform(double par, int seed); 
int sample_label(int doc_idx, int seed);
void evalute_classification();


int main() 
{ 
	run();
    return (0); 
}


void run()
{
	// load 测试集和训练集文件，从中初始化一部分文件级的全局变量 (文件级标签和wordcount) 
	int N1 = load_data_file("earn_acp_r8_train.txt", &wcArrTrain, &labelsTrain);
	int N2 = load_data_file("earn_acp_r8_test.txt", &wcArrTest, &labelsTest);
	// 文章总数 
	int N = N1 + N2;
	
	// 词库每个词的位置
	init_corpus(); 
	
	// 对每个词随机采样，初始化全局变量 
	init_training_variables();
	
	// 初始化 theta0 & theta1 , pseudo count = 1.0;
	int V = vocabulary.size();
	// V = 14308 in this case
	cout << "size of Vocabulary: " << V << endl;
	 
	// 初始化结果变量	
	cntCollection = 0;
	labelsTrainCollection.clear();
	for (int j = 0; j < N1; ++j) 
	{
		labelsTrainCollection.push_back(0); 
	}
	labelsTestCollection.clear();
	for (int j = 0; j < N2; ++j) 
	{
		labelsTestCollection.push_back(0); 
	}
	
	
	// 初始化完毕，开始运行 MAX_ITERATION 个迭代 
	for (int loop = 0; loop < MAX_ITERATION; ++loop)
	{
		cout << "Iteration " << loop+1 << " is ongoing ..." << endl;
		
		// 现在训练集中做一次 
		for (int i = 0; i < N1; ++i)
		{
			int docLen = wlArrTrain[i].size();
			// 每个词都做一次处理 
			for (int j = 0; j < docLen; ++j)
			{
				int widx =  get_word_idx(wcArrTrain[i][j]);
				// Step1. 去掉对这个词的采样结果，更新全局变量 
				remove_word(wlArrTrain[i][j], i, widx);
				
				// Step2. Gibbs Sampling 
				double logpr0 = calculate_logpr(0, i, widx, V);
				double logpr1 = calculate_logpr(1, i, widx, V);
				// log(pr0/pr1) = logpr0 - logpr1  ==>  pr0/pr1 = exp(logpr0 - logpr1)  ==> 
				// pr0 / (pr0 + pr1) = 1 / ( 1 + exp(logpr1 / logpr0) ) 
				double normalized_p0 = 1. / (1. + exp(logpr1 - logpr0));
				wlArrTrain[i][j] = sample_uniform(normalized_p0, j) ? 0 : 1;
				
				// Step3. 加上对这个词的重新采样结果，更新全局变量 
				add_word(wlArrTrain[i][j], i, widx);  
			}
		}

		// 同样的事情，在测试集中再做一遍，当然这里训练集和测试集同等对待，unsupervised 
		for (int i = N1; i < N; ++i)
		{
			int docLen = wlArrTest[i - N1].size();
			for (int j = 0; j < docLen; ++j)
			{
				int widx =  get_word_idx(wcArrTest[i - N1][j]);
				remove_word(wlArrTest[i - N1][j], i, widx);
				
				double logpr0 = calculate_logpr(0, i, widx, V);
				double logpr1 = calculate_logpr(1, i, widx, V);
				double normalized_p0 = 1. / (1. + exp(logpr1 - logpr0));
				wlArrTest[i - N1][j] = sample_uniform(normalized_p0, j) ? 0 : 1;
				
				add_word(wlArrTest[i - N1][j], i, widx);  
			}
		}
		
		// 一次 iteration 完成，检查是否记录本次的结果
		if (loop >= BURN_IN && loop % LAG == 0)
		{
			cntCollection ++;
			for (int j = 0; j < N1; ++j) 
			{
				labelsTrainCollection[j] += sample_label(j, j); 
			}
			for (int j = N1; j < N; ++j) 
			{
				labelsTestCollection[j - N1] += sample_label(j, j); 
			}
		} 
	}
	
	evalute_classification();
}


/*
vector<vector<string> > *pWCArr   : 每个个文件的信息，信息为词数组    
vector<int> *pLabels              : 每个位置保存一个文件的信息，信息为该文件对应的 label (0/1) 

本文件还顺便初始化了词库 vocabulary 
*/
int load_data_file(char *filename, vector<vector<string> > *pWCArr, vector<int> *pLabels)
{	
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
			}
			else
			{
				pLabels->push_back(1);
			}
			
			// update word account
			vector<string> vec2;
			str_explode(vec1[1].c_str(), ' ', vec2);
			int vec2len = vec2.size();
			for (int i = 0; i < vec2len; ++i)
			{
				// 加入词典 
				vocabulary.insert(vec2[i]);
			}
			pWCArr->push_back(vec2); 
		}
	}

	ifsm.close();
	ifsm.clear();	
	return pLabels->size();
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

// 根据已经在 load_data_file 中得到的  vocabulary
// 来初始化每个词的位置 
void init_corpus()
{
	int i = 0;
	for (set<string>::iterator it = vocabulary.begin(); it != vocabulary.end(); ++it)
	{
		vocLoc.insert(map<string, int>::value_type(*it, i));
		i ++;
	} 
} 


// 去掉一个词采样的结果 
// label   : 采样结果
// doc_idx : 词所在的doc
// word_idx: 词在词库中的位置 
void remove_word(int label, int doc_idx, int word_idx)
{
	nmttl[doc_idx] --;
	if (label == 0)
	{
		nm0[doc_idx] --;
		nk0[word_idx] --;
		nkttl0 --;
	}
	else
	{
		nm1[doc_idx] --;
		nk1[word_idx] --;
		nkttl1 --;
	}
}

// 加入一个词采样的结果 
void add_word(int label, int doc_idx, int word_idx) 
{
	nmttl[doc_idx] ++;
	if (label == 0)
	{
		nm0[doc_idx] ++;
		nk0[word_idx] ++;
		nkttl0 ++;
	}
	else
	{
		nm1[doc_idx] ++;
		nk1[word_idx] ++;
		nkttl1 ++;
	}
}


// 根据当前全局变量的情况来计算概率 
// vocLen: 词库词度
double calculate_logpr(int label, int doc_idx, int word_idx, int vocLen)
{
	double pr = 0.0;
	if (label == 0)
	{
		// 通过整个词空间中词的采样分布来衡量其概率 
		pr += log(nk0[word_idx] + BETA) -  log(nkttl0 + BETA * vocLen);
		// 通过文章 doc 本身的词分类分步来衡量某个词采样的概率；由于共两个分类，0 & 1，故此乘以 2 
		pr += log(nm0[doc_idx] + ALPHA) - log(nmttl[doc_idx] + ALPHA * 2 - 1);
	} 
	else
	{
		pr += log(nk1[word_idx] + BETA) -  log(nkttl1 + BETA * vocLen);
		// 通过文章 doc 本身的词分类分步来衡量某个词采样的概率；由于共两个分类，0 & 1，故此乘以 2 
		pr += log(nm1[doc_idx] + ALPHA) - log(nmttl[doc_idx] + ALPHA * 2 - 1);
	} 
	return pr;
} 
 
/*
初始化，随机分配每个词的几率
*/ 
void init_training_variables()
{
	// 训练集文章数 N1
	int nTrain = labelsTrain.size();
	// 测试集文章数 N2
	int nTest  = labelsTest.size(); 

	// 初始化全局变量
	nm0.clear(); nm1.clear(); nmttl.clear(); nk0.clear(); nk1.clear(); wlArrTrain.clear(); wlArrTest.clear(); 
	for (int i = 0; i < nTrain + nTest; ++i)
	{
		nm0.push_back(0);
		nm1.push_back(0);
		nmttl.push_back(0); 
	}
	int wcnt = vocabulary.size();
	for (int i = 0; i < wcnt; ++i)
	{
		nk0.push_back(0);
		nk1.push_back(0);
	}
	nkttl0 = 0;
	nkttl1 = 0;


	// 词随机采样。每个词操作一次，更新每个文章采样为topic 0/1的词的数量、更新每个词被采样为topic0/1的总次数 
	srand(time(0));
	for (int i = 0; i < nTrain; ++i)
	{
		int docLen = wcArrTrain[i].size();
		nmttl[i] = docLen;
		vector<int> labels;
		labels.clear();
		
		for (int j = 0; j < docLen; ++j)
		{
			int widx =  get_word_idx(wcArrTrain[i][j]);
			if (rand() % 2 == 0)
			{
				nm0[i] ++;
				nk0[widx] ++;
				nkttl0 ++;
				labels.push_back(0);
			}
			else
			{
				nm1[i] ++;
				nk1[widx] ++;
				nkttl1 ++;
				labels.push_back(1);
			}
		}
		
		wlArrTrain.push_back(labels);
		labels.clear();
	}
		
	// 测试数据接在训练数据后面，正如 header 部分所说，本代码不区分测试数据还是训练数据，当做 unsupervised 来训练 
	for (int i = nTrain; i < nTrain + nTest; ++i)
	{
		int docLen = wcArrTest[i - nTrain].size();
		nmttl[i] = docLen;
		vector<int> labels;
		labels.clear();
		
		for (int j = 0; j < docLen; ++j)
		{
			int widx =  get_word_idx(wcArrTest[i - nTrain][j]);
			if (rand() % 2 == 0)
			{
				nm0[i] ++;
				nk0[widx] ++;
				nkttl0 ++;
				labels.push_back(0);
			}
			else
			{
				nm1[i] ++;
				nk1[widx] ++;
				nkttl1 ++;
				labels.push_back(1);
			}
		}
		
		wlArrTest.push_back(labels);
		labels.clear();
	}

}

// 输入一个词，输出它在 vecLoc 中的位置 
int get_word_idx(string s)
{
	map<string, int>::iterator it = vocLoc.find(s);
	if (it != vocLoc.end())
	{
		return it->second;
	} 
	else
	{
		cout << "WRONGGGGG, never saw | " << s << " | before !" << endl; 
		return -1;
	}
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


// 根据当前全剧变脸的情况来计算某文档的分类 
int sample_label(int doc_idx, int seed)
{
	double theta0 = (double)(nm0[doc_idx] + ALPHA) / (nmttl[doc_idx] + ALPHA * 2); 
	double theta1 = (double)(nm1[doc_idx] + ALPHA) / (nmttl[doc_idx] + ALPHA * 2); 
	return sample_uniform(theta0 / (theta0 + theta1), seed) ? 0 : 1;
}


// 对算法所得到的结果进行评估 
// 注意，由于 unsupervised，故此得到的两个分类，有可能和原来的 0/1 完全相反，那么不能直接比较 0/1 分类中的文件 (dataset中的分类0可能是得到的分类1)
// 故此，应该检查相同分类的doc是否被分类到了一起 
void evalute_classification ()
{
	int N1 = labelsTrainCollection.size(); 
	for (int i = 0;  i < N1; ++i)
	{
		if ((double)labelsTrainCollection[i] / cntCollection >= 0.5)
		{
			labelsTrainCollection[i] = 1;
		}
		else 
		{
			labelsTrainCollection[i] = 0;
		}
	}
	
	int N2 = labelsTestCollection.size(); 
	for (int i = 0;  i < N2; ++i)
	{
		if ((double)labelsTestCollection[i] / cntCollection >= 0.5)
		{
			labelsTestCollection[i] = 1;
		}
		else 
		{
			labelsTestCollection[i] = 0;
		}
	}

	int correct = 0;
    // 最简单的方法，是正着比一遍，反着比一遍，取比较好的那个结果 
    // 正着比
   	for (int i = 0;  i < N1; ++i)
	{
		if (labelsTrainCollection[i] == labelsTrain[i])
			correct ++;
	}
   	for (int i = 0;  i < N2; ++i)
	{
		if (labelsTestCollection[i] == labelsTest[i])
			correct ++;
	}
	cout << "RESULT: " << correct << " correct out of " << N1 + N2 << " samples." << endl;
	// 反着比，结果应该正相反
	cout << "    OR: " << N1 + N2 - correct << " correct out of " << N1 + N2 << " samples." << endl;
}



