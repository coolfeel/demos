#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define LIST_INIT_SIZE 100
#define LISTINCREMENT 10

typedef struct list {
	int *elem;  //�׵�ַ 
	int length; //˳����� 
	int listsize; //�ѷ���ĳ��� 
}arrayList,*pArrayList;


bool initList(pArrayList list);    //��ʼ˳��� 
bool insertElem(pArrayList list,int i,int e); //����Ԫ�� 
void listElems(pArrayList list); //���˳��� 
bool deleteElem(pArrayList list,int i,int *val); //ɾ��Ԫ�� 

int main() {  //������ 
	pArrayList list=NULL; //ֻ�Ƕ���һ���ṹ��ָ�룬���ָ��� 
	int i;
	int len; //��Ҫ��˳����� len
	int num; //�����Ԫ�� 
	int n; //ɾ��Ԫ��λ�� 
	int *val=NULL;
	
	arrayList l;  //����ṹ����� 
	list=&l; //���ṹ��ָ��ָ��˽ṹ�� 
	
	if (initList(list)) {
		printf("�����ɹ�\n");
		 
		printf("��������Ҫ��˳�����:\n");
		scanf("%d",&len);
			
		for (i=1;i<=len;i++) {
			scanf("%d",&num);
			insertElem(list,i,num); 
		}
		
		printf("���Ԫ��:");
		listElems(list);
		
		printf("\n");
		
		printf("������ɾ��Ԫ�ص�λ��:\n");
		scanf("%d",&n);
	
		deleteElem(list,n,val);
		
		printf("���Ԫ��:");
		listElems(list);
		
	}		
	else
		printf("����ʧ��");	
		
	return 0;
}

bool initList(pArrayList L) { //��ʼ˳��� 
	L->elem=(int *)malloc(LIST_INIT_SIZE*sizeof(int));
	if (!L->elem)
		return false;
	L->length=0;
	L->listsize=LIST_INIT_SIZE;
	return true;
} 

bool insertElem(pArrayList L,int i,int e) { //����Ԫ�� 
	int *newbase,*p,*q;
	
	if (i<1||i>L->length+1)
		return false;
		
	if (L->length>=L->listsize) {
		newbase=(int *)realloc(L->elem,(L->listsize+LISTINCREMENT)*sizeof(int));
		if (!newbase)
			return false;
		L->elem=newbase;
		L->listsize+=LISTINCREMENT;
	}
	
	q=&L->elem[i-1];  //qָ��Ҫ���Ԫ�ص�λ�� 
	
	for (p=&L->elem[L->length-1];p>=q;p--) {  //ͨ��pָ�뽫Ԫ������ƶ� 
		*(p+1)=*p;
	}
	
	*q=e;
	L->length++;
	
	return true;
} 

void listElems(pArrayList L) { //���ѭ��� 
	int i;
	for (i=0;i<L->length;i++)
		printf("%d",L->elem[i]);  //˳���Ԫ���±��0��ʼ 
}

bool deleteElem(pArrayList L,int i,int *val) {
	int *q;
	
	if (i<1||i>L->length)
		return false;
	q=&L->elem[i-1]; //qָ��ɾ��Ԫ�ص�λ�� 
	val=q; 
	printf("ɾ����Ԫ��Ϊ:%d ",*val);
	
	for (q++;q<=&L->elem[L->length-1];q++) { //��Ҫɾ����λ����һ����ʼ�ƶ� 
		*(q-1)=*q;
	}	
	L->length--;
	return true;
}
