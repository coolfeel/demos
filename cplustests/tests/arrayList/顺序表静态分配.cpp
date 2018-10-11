#include <stdio.h>
#define MaxSize 50

typedef struct array {
	int data[MaxSize];             //��С 
	int length;					  //��ǰ���� 
}Sqlist;

bool insert(Sqlist &l,int i,int e); //�����˳�������޸� ��ȡ��ַ& 
void print(Sqlist l);               //���޸Ŀ�ֱ�� ���ṹ����� 
int locateElem(Sqlist l,int e);

int main() {
	
	int i;
	Sqlist sql1;
	sql1.length=0;                 //init length 0 
	
	for (i=1;i<=10;i++) 
		insert(sql1,i,i);
	
	printf("����%d  ",sql1.length);
	
	for (i=0;i<sql1.length;i++)
		printf("%d ",sql1.data[i]);
	
	printf("λ��%d",locateElem(sql1,10));
	return 0;
}

bool insert(Sqlist &l,int i,int e) {
	int j;
	if (i<1||i>l.length+1)         //�ڵ�1��length+1��λ�ò��� 
		return false;
	if (l.length>=MaxSize)         //�жϵ�ǰ�Ƿ񳬹��������󳤶� 
		return false;
	for (j=l.length;j>=i;j--)     //�����һ����i������� 
		l.data[j]=l.data[j-1];
	l.data[i-1]=e;                //Ԫ�ز��� 
	l.length++;
	
	return true;
} 

void print(Sqlist l) {              //��� 
	int i;
	for (i=0;i<l.length;i++)
		printf("%d ",l.data[i]);
}

int locateElem(Sqlist l,int e) {     //��ֵ����λ�� 
	int i;
	for (i=0;i<l.length;i++) {
		if (l.data[i]==e) {
			return i+1;
		}
	}
	return 0;
}
