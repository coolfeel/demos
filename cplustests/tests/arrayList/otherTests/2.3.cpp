#include <stdio.h>
/*
����Ϊn��˳���L����дһ��ʱ�临�Ӷ�ΪO(n)���ռ临�Ӷ�ΪO(1),���㷨�����㷨ɾ�����Ա�
������ֵΪx��Ԫ�� 
*/ 

typedef struct arrayList {
	int *data;
	int maxSize,length;
}sqList;

int main() {
	
	return 0;
} 

void deleteEle(sqList &l,int x) {
	int *p,i;
	for (i=0;i<l.length;i++) {		
		if (l.data[i]==x) {		
			for (p=&(l.data[i]);p<&(l.data[l.length]);p++)
				*(p-1)=*p;
		}
	}	
}

void deleteElem(sqList &l,int x) {       //ֱ�ӰѲ���x�����鵽L��  
	int k=0;
	for (int i=0;i<l.length;i++) {
		if (l.data[i]!=x) {
			l.data[k]=l.data[i];
			k++;
		}	
	}
	l.length=k;
}
