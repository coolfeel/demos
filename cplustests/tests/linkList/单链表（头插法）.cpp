#include <stdio.h>
#include <stdlib.h>

typedef struct lNode {
	int data;
	struct lNode *next; 
}lNode,*linkedList;

linkedList initList(linkedList &l);            //��ʼ�� 
void print(linkedList l);                      //��� 
int getElem(linkedList l,int i);				//����Ż��Ԫ�� 
int locateElem(linkedList l,int e);             //��λԪ�� 

int main() {
	lNode l;
	linkedList p=&l;
	initList(p);
	print(p);
	printf("λ��Ԫ���ǣ�%d  ",getElem(p,1));
	printf("3��λ��:%d",locateElem(p,3));	
	return 0;
}

linkedList initList(linkedList &l) {
	int x;
	linkedList s;
	l=(linkedList)malloc(sizeof(lNode));
	l->next=NULL;
	
	scanf("%d",&x);
	while (x!=9999) {
		s=(linkedList)malloc(sizeof(lNode));
		s->data=x;
		s->next=l->next;
		l->next=s;
		scanf("%d",&x);
	}
	return l;
}

int getElem(linkedList l,int i) {
	int j=1;
	linkedList p=l->next;
	if (i<1)
		return 0;
	while (p&&j<i) {
		p=p->next;
		j++;
	}
	
	return p->data;
}

int locateElem(linkedList l,int e) {
	linkedList p=l->next;
	int i=1;
	while (p&&p->data!=e) {
		i=i+1;
		p=p->next;
	}
	return i;
}


void print(linkedList l) {
	linkedList p=l->next;
	while (p) {
		printf("%d ",p->data);
		p=p->next;
	}
} 
