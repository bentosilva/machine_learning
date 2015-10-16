/*
toy data��10��ѧϰ����ѧϰ������1��Σ����Ԥ����10��������˳�򻹶��ԣ�������Ȩ��
	w[0] = -64
    w[1] = -298
    w[2] = -360

˵������������߼��У��ҳ��ǵ�һλ�ģ��������������ɿ�����������

       ����    ����    �ҳ�    ����
�����  2       5       5       1
����    5       4       5       2
��ͳ    2       5       4       3
�ŷ�    5       3       5       4
��    5       4       4       5
κ��    5       4       4       6
�ܲ�    3       3       4       7
�ϴ�    4       3       2       8
����    3       3       1       9
�ƺ�    2       2       2       10
*/

/*
output
1
2
3
4
6
6
7
8
9
10
w0: -64
w1: -298
w2: -360
b1: -3330
b2: -3096
b3: -3047
b4: -3013
b5: -2999
b6: -2567
b7: -1891
b8: -1467
b9: -1444

Ϊɶ������6�أ����ǿ�������κ�ӵ�������ʵ����ͬ�ģ�û������ 
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include "stdint.h"

using namespace std;

// 2       5       5       1, ǰ�����Ƿ���������һ���� rank 
void split(const char* str, const char split_char, vector<uint32_t>& str_v)
{
	const char* phead = str;
	const char* ptail = str;
	
	for(;(phead = ptail)!='\0';)
	{
		string os;
		char buf[2];
		buf[1] = '\0';
		// append char by char
		for(ptail = phead; *ptail != split_char && *ptail !='\0'&& *ptail != '\n'; ptail ++)
		{
			buf[0] = *ptail;
			os.append(buf);
		}
		
		if (*ptail == split_char)
		{
			str_v.push_back(atoi(os.c_str()));
			ptail ++;
		}
		else if (*ptail == '\0' || *ptail == '\n')
		{
			str_v.push_back(atoi(os.c_str()));
			return;
		}
		else
			return;
	}
}

/* i have another split implementation
void CStat4Utils::strExplode(const TCHAR *pStr, TCHAR cSep, std::vector<std::string>& vecItems)
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
*/

int main(int argc, char** argv) {
	// load file
	FILE *fp;
	char line[20];
	size_t len = 0;
//	ssize_t read;
	
	// test.txt 
	fp = fopen("test.txt", "r");
	if (fp == NULL)
		exit(1);
	
	vector< vector<uint32_t> > map;
	while (fgets(line, 20, fp))
	{
//		line[read-1] = '\0';
		vector<uint32_t> u_v;
		split(line, '\t', u_v);
		map.push_back(u_v);
	}
	fclose(fp);
	
	// init w
	vector<float> w;
	w.push_back(0); w.push_back(0); w.push_back(0);
	
	// init b, ��ԭ���Ӳ�ͬ��Ӧ���� b1 ~ b9 Ϊ 0�� b10 Ϊ infinite; 
	// ԭ����b10 ҲΪ0�� b11 Ϊ infinite 
	float b[11] = {-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000000};
	float index[11];
	float tao[11];
	
	// iteration 1,000,000
	for (int iter = 0; iter < 1000000; ++iter)
	{
		// for each iteration, iterate all the items
		for (int i =0; i < map.size(); ++i)
		{
			// predict rank ,init as the smallest one, 1
			int predict_r = 1;
			float inner_p = w[0]*map[i][0] + w[1]*map[i][1] + w[2]*map[i][2];
			
			for (int r = 1; r <= 10; ++r)
			{
				// �ҵ�Ԥ�ڵ�rank������˵��һ��λ�� r��ʹ�� w.x < br 
				if (inner_p - b[r] < 0)
				{
					predict_r = r;
					break;
				}
			}
			
			int real_r = map[i][3];
			
			// rank��ͬ������Ҫ�������� w �� b 
			if (real_r != predict_r)
			{
				for (int r = 1; r <= 10; ++r)
				{
					// ע�⣬���� r=real_r,�õ�Ϊ -1
					// �����ģ����� index = 1 �ĵ㣬���� y - 1 
					if (real_r <= r)
					{
						index[r] = -1;
					}
					else
					{
						index[r] = 1;
					}
				} 
				float tao_sum = 0.0;
				for (int r = 1; r <= 10; ++r)
				{
					if ((inner_p - b[r]) * index[r] <= 0)
					{
						tao[r] = index[r];
					} 
					else
					{
						tao[r] = 0;
					}
					tao_sum += tao[r];
				}
				
				// ��ʼ����
				// w = w + sum(y(r)).x
				w[0] = w[0] + tao_sum * map[i][0];
				w[1] = w[1] + tao_sum * map[i][1];
				w[2] = w[2] + tao_sum * map[i][2];
				for (int r = 1; r <= 10; ++r)
				{
					// b(r) = b(r) - y(r)
					b[r] = b[r] - tao[r];
				} 
			} 
		}	
	}
	
	// �㷨������ϣ��õ����� w �� b�����Կ�ʼ������
	// ʹ��ԭ���ݼ����ȷ��
	for (int i = 0; i < map.size(); ++i)
	{
		int predict_r = 1;
		float inner_p = w[0]*map[i][0] + w[1]*map[i][1] + w[2]*map[i][2];
		for (int r = 1; r <= 10; r++)
		{
			if (inner_p - b[r] < 0)
			{
				predict_r = r;
				break;
			}
		}
		cout << predict_r << endl;
	} 
	cout << "w0: " << w[0] << endl;
	cout << "w1: " << w[1] << endl;
	cout << "w2: " << w[2] << endl;
	
	// b10 is infinite 
	cout << "b1: " << b[1] << endl;
	cout << "b2: " << b[2] << endl;
	cout << "b3: " << b[3] << endl;
	cout << "b4: " << b[4] << endl;
	cout << "b5: " << b[5] << endl;
	cout << "b6: " << b[6] << endl;
	cout << "b7: " << b[7] << endl;
	cout << "b8: " << b[8] << endl;
	cout << "b9: " << b[9] << endl;

	return 0;
}

