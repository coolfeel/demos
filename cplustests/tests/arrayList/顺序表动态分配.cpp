#include <stdio.h>
#include <stdlib.h>

#define InitSize 100  //��ʼ���� 
#define INCREMENT 50  //���� 

typedef struct array {
	int *data;
	int length,MaxSize;
}Sqlist;


void print(Sqlist l); //��� 
bool init(Sqlist &l);	//��ʼ�� 
void destory(Sqlist &l); //���� 
void isEmpty(Sqlist l); //�п� 
int getLength(Sqlist l); //��ó��� 
int locateElem(Sqlist l,int e); //��λ�ض�Ԫ�� 
void nextElem(Sqlist l,int e); //��һ��Ԫ�� 
int getElem(Sqlist l,int i,int *e); //�õ�Ԫ�� 
bool insert(Sqlist &l,int i,int e); //���� 
void deleteElem(Sqlist &l,int i,int *e);  //ɾ��      


int main() {
	
	int e=123;
	
	Sqlist sql1;
	
	init(sql1);
	
	insert(sql1,1,222);
	insert(sql1,2,22);
	insert(sql1,3,2224);
	
	deleteElem(sql1,1,&e);
	printf("ɾ������%dԪ��  ",e);
	print(sql1);
	
	
	return 0;
}

bool init(Sqlist &l) {                   //��ʼ�� 
	l.data=(int *)malloc(sizeof(int)*InitSize);  //��̬���� 
	if (l.data==NULL)
		return false;
	l.length=0;                        //���ȳ�ʼ��Ϊ0 
	l.MaxSize=InitSize;              //��ǰ���������� 
	return true;
} 

void destory(Sqlist &l) {           //���� 
	if (l.data!=NULL) {              //�����ڴ� ֻ���ͷ��׵�ַ 
		free(l.data); 
		printf("˳���������");
	}
		
} 

void isEmpty(Sqlist l) {             //�п� 
	if (l.data!=NULL) {
		if (l.length==0) {            //������ 
			printf("������Ԫ��");
		}
		else
			printf("����Ԫ��"); 
	}
	else
		printf("������");
} 


int getLength(Sqlist l) {
	if (l.data!=NULL)                    //����ڣ����س��� 
		return l.length;
	else
		printf("������");
	return 0;
}


int locateElem(Sqlist l,int e) {         //�ҵ��ض�Ԫ��λ�� 
	int i;
	for (i=0;i<l.length;i++) {
		if (l.data[i]==e) {
			printf("Ԫ�ص�λ��Ϊ%d",i+1);
			return i+1;
		} 
	}
	printf("û�д�Ԫ��");
	return 0;
} 


void nextElem(Sqlist l,int e) {           //���ض�Ԫ�غ��Ԫ�� 
	int i;
	i=locateElem(l,e);
	if (i!=0) {
		if (i=l.length)
			printf("���һ��Ԫ��û�к��");
		else
			printf("���Ϊ%d",l.data[i]);
	}
	else
		printf("û�д�Ԫ��");
}


int getElem(Sqlist l,int i,int *e) {      //�õ�ȷ��λ��Ԫ�� 
	if (i<1||i>l.length) 
		return 0; 
	*e=l.data[i-1];
	return *e;
}


void clear(Sqlist &l) {                   //����Ϊ0����� 
	if (l.data!=NULL)
		l.length=0;
}



bool insert(Sqlist &l,int i,int e) {
	
	int *newbase;               //�·�����׵�ַ 
	int *p,*q;
	
	if (i<1||i>l.length+1)
		return false;
		
	if (l.length>=l.MaxSize) {
		newbase=(int *)realloc(l.data,(l.MaxSize+INCREMENT)*sizeof(int));
		if (newbase==NULL)
			return false;
		else
			l.data=newbase;
	}
	
	p=&(l.data[i-1]);                             //pָ�����λ�� 
	
	for (q=&(l.data[l.length-1]);q>=p;q--)        //�����λ�ÿ�ʼ������λ������� 
		*(q+1)=*q;

	*p=e;                                         //eԪ�ز��� 
	l.length++;
	
	return true;
}


void deleteElem(Sqlist &l,int i,int *e) {     //ɾ��Ԫ��,ɾ����Ԫ��ȡָ�룬�ں����ⲿҲ�ܷ��� 
											  //�����βο����޸�ʵ�� 
	int *p;
	
	if (i<1||i>l.length)
		return;
	p=&(l.data[i-1]);                       //pָ��ɾ��λ�� 
	*e=l.data[i-1];                          
	 
	for (;p<&(l.data[l.length-1]);p++)      //p֮�����ǰ�ƶ��������������ڶ���Ԫ�ؾ��� 
		*p=*(p+1);                           //p+1��ָ���һ���� 
	
	l.length--;
} 

void print(Sqlist l) {
	int i;
	for (i=0;i<l.length;i++) {
		printf("%d ",l.data[i]);
	}
}
