#include <stdio.h>
//���һ����Ч���㷨����˳����Ԫ������Ҫ��ռ临�Ӷ�ΪO(1)
 
typedef struct arrayList {
	int *data;
	int maxSize,length;	
}sqList;

int main() {
	return 0;
} 

void reverse (sqList &l) {
	int t;
	for (int i=0;i<l.length/2;i++) {
		t=l.data[i];
		l.data[i]=l.data[l.length-1-i];
		l.data[l.length-1-i]=t;
	}
		
}


void reverseElem (sqList &l) {
	int temp;                     //ֻ����һ�������������� 
	for (int i=0;i<l.length/2;i++) {
		temp=l.data[i];
		l.data[i]=l.data[l.length-1-i];
		l.data[l.length-1-i]=temp;
	}
}
