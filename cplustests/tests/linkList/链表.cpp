#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

typedef struct Node {
	int data;
	struct Node *pNext;
}NODE,*PNODE; 

	
PNODE create_list();
void traverse_list(PNODE pHead);
bool is_empty(PNODE pHead); 
int length_list(PNODE);
bool insert_list(PNODE,int,int);
bool delete_list(PNODE,int,int *);
void sort_list(PNODE);


int main() {
	PNODE pHead=NULL;
	pHead=create_list();
	traverse_list(pHead);
	return 0;
}


PNODE create_list() {
	int len;								//��Ч�ڵ㳤�� 
	int i;
	int val;								//�����ʱ����Ľڵ�ֵ
	
	PNODE pHead=(PNODE)malloc(sizeof(NODE));
	if (pHead==NULL) {
		printf("�ڴ����ʧ��");
		exit(-1);
	}
	
	PNODE pTail=pHead;                     //���õ�ǰβ�ڵ� 
	pTail->pNext=NULL; 
	
	printf("������ڵ������len=");
	scanf("%d",&len);
	
	for (i=0;i<len;i++) {
		printf("�������%d���ڵ��ֵ:",i+1);
		scanf("%d",&val);
		
		PNODE pNew=(PNODE)malloc(sizeof(NODE));
		
		if (pNew==NULL) {
			printf("�ڴ����ʧ��");
			exit(-1);
		}
		
		//pNew->data=val;
		//pHead->pNext=pNew;      ȫ����ͷָ���½ڵ� 
		//pNew->pNext=NULL; 
		
		pNew->data=val;           
		pTail->pNext=pNew;        //pTail�ǵ�ǰβ�ڵ� 
		pNew->pNext=NULL;
		pTail=pNew; 
	} 
	
	return pHead;
}


void traverse_list(PNODE pHead) {
	PNODE p=pHead->pNext;
	while(p!=NULL) {
		printf("%d",p->data);
		p=p->pNext;
	}
	printf("\n");
}


bool is_empty(PNODE pHead) {
	if (pHead->pNext==NULL) 
		return true;
	else 
		return false;
}


int length_list(PNODE pHead) {
	PNODE p=pHead->pNext;
	int len=0;
	while(p!=NULL) {
		len+=1;
		p=p->pNext;
	}
	return len;
}
