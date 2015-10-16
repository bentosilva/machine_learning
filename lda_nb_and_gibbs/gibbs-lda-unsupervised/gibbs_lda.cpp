/************************************************************************
Gibbs LDA

File earn_acp_r8_train.txt -  earn count: 2840    acq count: 1596
File earn_acp_r8_test.txt -  earn count: 1083    acq count: 696
size of Vocabulary: 14308

����汾������� Unsupervised ѧϰ������ѵ�����Ͳ��Լ������֣�����ͬ�ȿ���������
��������������������е��߼���ȫһ�� 
���õ��Ľ������֪����Ƚ�

���ڲ����˸��򵥵����ݽṹ���ʴ����б� Gibbs Naive Bayes ���� 
 
-----
Iteration 10 | burn in 3 | lap 2 Result:   (Pretty Bad)
	
RESULT: 3786 correct out of 6215 samples.  (ȡ���)
    OR: 2429 correct out of 6215 samples.
    
-----
Iteration 50 | burn in 15 | lap 5 Result:   (Better)
	
RESULT: 1980 correct out of 6215 samples.
    OR: 4235 correct out of 6215 samples.  (ȡ���)
    
-----
Iteration 500 | burn in 100 | lap 8 | ALPHA 25 | BETA 0.01  Result:   (Much Better)

RESULT: 5285 correct out of 6215 samples.  (ȡ���) 
    OR: 930 correct out of 6215 samples.
    
-----
Iteration 500 | burn in 100 | lap 8 | ALPHA 25 | BETA 1.  Result:   (not change much)
RESULT: 5381 correct out of 6215 samples.  (ȡ���) 
    OR: 834 correct out of 6215 samples.    

ֱ���˴������еĽ����û�� gibbs_naivebayes �Ľ����
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
#define ALPHA 25	// alpha  50/K = 50/2 = 25���������� 
#define BETA  0.01  // beta 0.01
#define MAX_BIG_BUFF_SIZE 50000 //

// ȫ�ֱ��� - �������ļ��л�ȡ����ʵ��Ϣ 
vector<vector<string> > wcArrTrain; 	// TRAIN dataset ÿ�����µ� word ����
vector<vector<string> > wcArrTest;		// TEST  dataset ÿ�����µ� word ����
vector<int> labelsTrain;	// TRAIN dataset ÿ�����µ� label ����
vector<int> labelsTest; 	// TEST  dataset ÿ�����µ� label ����

set<string> vocabulary;		// �����ʿ⣬set������ģ��������ִʿ��˳��Ҳ�ͱ����� wordcount ��ÿ�����ڴʿ��λ�� 
map<string, int> vocLoc;	// �����ʿ⣬ÿ���ʺ�����λ�� 

// ȫ�ֱ��� - �����������б����㱻���� 
vector<vector<int> > wlArrTrain; 		// TRAIN dataset ÿ�����µ� label���� (ÿ���ʵ�label)
vector<vector<int> > wlArrTest;			// TEST  dataset ÿ�����µ� label���� (ÿ���ʵ�label) 
vector<int> nm0;	// ÿ��Ԫ���Ƕ�Ӧ���²���Ϊ topic 0 �Ĵ�������ά��Ϊ�����ܸ��� 
vector<int> nm1;	// ÿ��Ԫ���Ƕ�Ӧ���²���Ϊ topic 1 �Ĵ����� 
vector<int> nmttl;	// ÿ��Ԫ���Ƕ�Ӧ���µĴ�������ά��Ϊ�����ܸ���;   nm0 / nmttl = doc m Ϊ topic 0 �ļ��� (�����topic 1)
vector<int> nk0;	// ÿ��Ԫ���Ƕ�Ӧ�ʲ���Ϊ topic 0 ���ܴ�����ά��Ϊ�ʿ��дʸ��� 
vector<int> nk1;	// ÿ��Ԫ���Ƕ�Ӧ�ʲ���Ϊ topic 1 ���ܴ��� 
int nkttl0;			// ���д��в���Ϊ topic 0 ���ܴ��� 	  nk0 / nkttl0 Ϊ �� k �� topic 0 �µļ��� (�����������) 
int nkttl1; 		// ���д��в���Ϊ topic 1 ���ܴ���;   

// ȫ�ֱ��� - �ռ�������������ڼ����㷨
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
	// load ���Լ���ѵ�����ļ������г�ʼ��һ�����ļ�����ȫ�ֱ��� (�ļ�����ǩ��wordcount) 
	int N1 = load_data_file("earn_acp_r8_train.txt", &wcArrTrain, &labelsTrain);
	int N2 = load_data_file("earn_acp_r8_test.txt", &wcArrTest, &labelsTest);
	// �������� 
	int N = N1 + N2;
	
	// �ʿ�ÿ���ʵ�λ��
	init_corpus(); 
	
	// ��ÿ���������������ʼ��ȫ�ֱ��� 
	init_training_variables();
	
	// ��ʼ�� theta0 & theta1 , pseudo count = 1.0;
	int V = vocabulary.size();
	// V = 14308 in this case
	cout << "size of Vocabulary: " << V << endl;
	 
	// ��ʼ���������	
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
	
	
	// ��ʼ����ϣ���ʼ���� MAX_ITERATION ������ 
	for (int loop = 0; loop < MAX_ITERATION; ++loop)
	{
		cout << "Iteration " << loop+1 << " is ongoing ..." << endl;
		
		// ����ѵ��������һ�� 
		for (int i = 0; i < N1; ++i)
		{
			int docLen = wlArrTrain[i].size();
			// ÿ���ʶ���һ�δ��� 
			for (int j = 0; j < docLen; ++j)
			{
				int widx =  get_word_idx(wcArrTrain[i][j]);
				// Step1. ȥ��������ʵĲ������������ȫ�ֱ��� 
				remove_word(wlArrTrain[i][j], i, widx);
				
				// Step2. Gibbs Sampling 
				double logpr0 = calculate_logpr(0, i, widx, V);
				double logpr1 = calculate_logpr(1, i, widx, V);
				// log(pr0/pr1) = logpr0 - logpr1  ==>  pr0/pr1 = exp(logpr0 - logpr1)  ==> 
				// pr0 / (pr0 + pr1) = 1 / ( 1 + exp(logpr1 / logpr0) ) 
				double normalized_p0 = 1. / (1. + exp(logpr1 - logpr0));
				wlArrTrain[i][j] = sample_uniform(normalized_p0, j) ? 0 : 1;
				
				// Step3. ���϶�����ʵ����²������������ȫ�ֱ��� 
				add_word(wlArrTrain[i][j], i, widx);  
			}
		}

		// ͬ�������飬�ڲ��Լ�������һ�飬��Ȼ����ѵ�����Ͳ��Լ�ͬ�ȶԴ���unsupervised 
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
		
		// һ�� iteration ��ɣ�����Ƿ��¼���εĽ��
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
vector<vector<string> > *pWCArr   : ÿ�����ļ�����Ϣ����ϢΪ������    
vector<int> *pLabels              : ÿ��λ�ñ���һ���ļ�����Ϣ����ϢΪ���ļ���Ӧ�� label (0/1) 

���ļ���˳���ʼ���˴ʿ� vocabulary 
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
				// ����ʵ� 
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

// �����Ѿ��� load_data_file �еõ���  vocabulary
// ����ʼ��ÿ���ʵ�λ�� 
void init_corpus()
{
	int i = 0;
	for (set<string>::iterator it = vocabulary.begin(); it != vocabulary.end(); ++it)
	{
		vocLoc.insert(map<string, int>::value_type(*it, i));
		i ++;
	} 
} 


// ȥ��һ���ʲ����Ľ�� 
// label   : �������
// doc_idx : �����ڵ�doc
// word_idx: ���ڴʿ��е�λ�� 
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

// ����һ���ʲ����Ľ�� 
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


// ���ݵ�ǰȫ�ֱ����������������� 
// vocLen: �ʿ�ʶ�
double calculate_logpr(int label, int doc_idx, int word_idx, int vocLen)
{
	double pr = 0.0;
	if (label == 0)
	{
		// ͨ�������ʿռ��дʵĲ����ֲ������������ 
		pr += log(nk0[word_idx] + BETA) -  log(nkttl0 + BETA * vocLen);
		// ͨ������ doc ����Ĵʷ���ֲ�������ĳ���ʲ����ĸ��ʣ����ڹ��������࣬0 & 1���ʴ˳��� 2 
		pr += log(nm0[doc_idx] + ALPHA) - log(nmttl[doc_idx] + ALPHA * 2 - 1);
	} 
	else
	{
		pr += log(nk1[word_idx] + BETA) -  log(nkttl1 + BETA * vocLen);
		// ͨ������ doc ����Ĵʷ���ֲ�������ĳ���ʲ����ĸ��ʣ����ڹ��������࣬0 & 1���ʴ˳��� 2 
		pr += log(nm1[doc_idx] + ALPHA) - log(nmttl[doc_idx] + ALPHA * 2 - 1);
	} 
	return pr;
} 
 
/*
��ʼ�����������ÿ���ʵļ���
*/ 
void init_training_variables()
{
	// ѵ���������� N1
	int nTrain = labelsTrain.size();
	// ���Լ������� N2
	int nTest  = labelsTest.size(); 

	// ��ʼ��ȫ�ֱ���
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


	// �����������ÿ���ʲ���һ�Σ�����ÿ�����²���Ϊtopic 0/1�Ĵʵ�����������ÿ���ʱ�����Ϊtopic0/1���ܴ��� 
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
		
	// �������ݽ���ѵ�����ݺ��棬���� header ������˵�������벻���ֲ������ݻ���ѵ�����ݣ����� unsupervised ��ѵ�� 
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

// ����һ���ʣ�������� vecLoc �е�λ�� 
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


// ���ݵ�ǰȫ����������������ĳ�ĵ��ķ��� 
int sample_label(int doc_idx, int seed)
{
	double theta0 = (double)(nm0[doc_idx] + ALPHA) / (nmttl[doc_idx] + ALPHA * 2); 
	double theta1 = (double)(nm1[doc_idx] + ALPHA) / (nmttl[doc_idx] + ALPHA * 2); 
	return sample_uniform(theta0 / (theta0 + theta1), seed) ? 0 : 1;
}


// ���㷨���õ��Ľ���������� 
// ע�⣬���� unsupervised���ʴ˵õ����������࣬�п��ܺ�ԭ���� 0/1 ��ȫ�෴����ô����ֱ�ӱȽ� 0/1 �����е��ļ� (dataset�еķ���0�����ǵõ��ķ���1)
// �ʴˣ�Ӧ�ü����ͬ�����doc�Ƿ񱻷��ൽ��һ�� 
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
    // ��򵥵ķ����������ű�һ�飬���ű�һ�飬ȡ�ȽϺõ��Ǹ���� 
    // ���ű�
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
	// ���űȣ����Ӧ�����෴
	cout << "    OR: " << N1 + N2 - correct << " correct out of " << N1 + N2 << " samples." << endl;
}



