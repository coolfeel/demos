#include <stdio.h>
#include <stdlib.h>
/*
	��˳�����ɾ��������Сֵ��Ԫ�أ�����Ψһ�����������ر�ɾ��Ԫ�أ��ճ���λ�������һ��Ԫ���
	��Ϊ��ʱ��ʾ������Ϣ 
*/
typedef struct arrayList {
	int *data;
	int maxSize,length;	
}sqList;

int main() {
	
	return 0;
}

int back(sqList &l,int *e) {
	
	int min,t;
	
	if (l.data==NULL) {                     //�׵�ַ�п� 
		printf("˳���Ϊ��");
		exit(1);
	}
	
	min=l.data[0];
	for (int i=1;i<l.length;i++) {
		if (l.data[i]<min) {
			min=l.data[i];
			t=i;
		}
	}
	l.data[t]=l.data[l.length-1];
	return min;                            //���û������ 
		
} 



bool deleteElem(sqList &l,elemType &value) {
	//ɾ��˳�������СֵԪ�ؽ�㣬��ͨ�������Ͳ���e������ֵ
	if (l.length==0)                    //��գ��ó����ж� 
		return false;
	value=l.data[0];
	int pos=0;
	for (int i=1;i<l.length;i++)      
		if (l.data[i]<value) {          //����̨�㷨 
			value=l.data[i];
			pos=i;                      //pos��ס��Сֵ�±� 
		} 
	l.data[pos]=l.data[l.length-1];
	l.length--;                         //���ȼ��� 
	return true;
}
