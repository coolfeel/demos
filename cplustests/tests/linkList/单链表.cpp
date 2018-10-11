#include <stdio.h>
#include <stdlib.h>

typedef struct lNode {
	int data;
	struct lNode *next;
}lNode,*linkList;

int main() {
	return 0;
}

linkList initList(linkList &L) {
	int x;
	linkList s;
	L=(linkList)malloc(sizeof(lNode));             //����ͷ��� 
	L->next=NULL;
	
	scanf("%d",&x);
	while (x!=0) {
		s=(linkList)malloc(sizeof(lNode));         //�����½��s 
		s->data=x;									//ͷ�巨,һֱ��ͷ���֮����� 
		s->next=L->next;
		L->next=s;
		scanf("%d",&x);
	}
	return L;
}
/*
linkList initList(linkList &L) {
	int x;
	L=(linkList)malloc(sizeof(lNode));            //����ͷ��� 
	L->next=NULL; 
	
	linkList s,r=L;                               //rΪβָ�룬ʼ��ָ������㣬��ʼָ��ͷ��� 
	scanf("%d",&x);
	while (x!=0) {
		s=(linkList)malloc(sizeof(lNode));       //�����½�㣬β�巨 
		r->next=s;
		r=s;
		scanf("%d",&x);
	}
	r->next=NULL;
	return L;
}
*/

linkList getElem(linkList L,int i) {            //����ŵõ�Ԫ�� ,��1��ʼ��0Ϊͷ��� 
	int j=1;
	linkList p=L->next;                         //pָ��ͷ������һ�� 
	if (i==0)
		return L;
	if (i<1)
		return NULL;
	while (p&&j<i) {                           //p����,jΪi-1ʱΪi��ǰһ��,p->nextΪiλ��Ԫ�� 
		p=p->next;
		j++;
	}
	return p;                                  //����iλ��Ԫ�ص�ַ 
}

int locateElem(linkedList L,int e) {           //��λԪ��λ�� 
	linkedList p=L->next;
	int i=1;
	while (p&&p->data!=e) {
		p=p->next;
		i=i+1;
	}
	if (p==NULL)                               //���������һ��������e����pΪNULL 
		return 0;
	else
		return i;
}

bool insert(linkList &L,int i,int e) {
	linkList s=(linkList)malloc(sizeof(lNode));
	if (s==NULL)
		return false;
	s->data=e;
	linkList p=getElem(L,i-1);                 //pָ��i��ǰһ��λ�� 
	s->next=p->next;
	p->next=s;
	return true;
}

bool deleteElem(linkList &L,int i,int &e) {
	linkList p=getElem(L,i-1);                 //pָ��i��ǰһ�� 
	linkList q=p->next;                        //qָ��Ҫɾ����i 
	e=q->data;
	p->next=q->next;
	free(q);
	return true; 
}
 
 
void destroyList(LinkList &L) {          //���ٵ����� 
����linkList p=L,q=p->next;              //pָ��ͷ��qָ��ͷ����һ�� 
����while (q!=NULL)
����{ 
��������free(p);
��������p=q;                             //pָ����һ��λ��q 
��������q=p->next;                       //qָ��p����һ�� 
����}
����free(p); /*��ʱqΪNULL, pָ��β���,�ͷ���*/
}

void print(linkList L) {
	linkList p=L->next;
	while (p) {
		printf("%d ",p->data);
		p=p->next;
	}
} 
