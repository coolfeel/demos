#include <stdio.h>
#include <stdlib.h>

#define MAXSIZE 100

typedef struct BiTNode {                    //��ʽ�洢�ṹ 
	char data;
	struct BiTNode *lchild,*rchild;
}BiTNode,*BiTree;


   
int visit(char c);                     //���ʽ�� 
int initBiTree(BiTree &T);             //��ʼ�� 
int createBiTree(BiTree &T);            //���� 
int isEmptyBiTree(BiTree T);           //�п� 
char getRoot(BiTree T);                 //��ø���� 
int depthBiTree(BiTree T);            //������ 
int destroyBiTree(BiTree &T);         //���� 
int preOrder(BiTree T);               //������� 
int inOrder(BiTree T);                //������� 
int lastOrder(BiTree T);             //������� 

int main() {
	
	BiTNode TN;
	BiTree T=&TN;
	
    	
    createBiTree(T);
    preOrder(T);
    printf("%d\n",depthBiTree(T));
    printf("%c\n",getRoot(T));
    inOrder(T);
    return 0;
}




int visit(char c) {             //���� 
	printf("%c",c);
	return 1;
} 


int initBiTree(BiTree &T) {      //��ʼ�� 
	T=NULL;
	return 1;
}


int createBiTree(BiTree &T) {    //���������� 
	char data;
	scanf("%c",&data);
	
	if (data=='#')
		T=NULL;
	else {
		T=(BiTNode *)malloc(sizeof(BiTNode));
		if (!T)
			return 0;
		T->data=data;
		createBiTree(T->lchild);
		createBiTree(T->rchild);
	}
	
}


int isEmptyBiTree(BiTree T) {   //�п� 
	if (!T)
		return 1;
	else
		return 0;
}


char getRoot(BiTree T) {           //�õ������ 
	return T->data;
}


int depthBiTree(BiTree T) {          //��������� 
	int i,j;
	if (!T)
		return 0;
	else {
		if (T->lchild)
			i=depthBiTree(T->lchild);
		else
			i=0;
		if (T->rchild)
			j=depthBiTree(T->rchild);
		else
			j=0;
	}
	return i>j?i+1:j+1;
}


int destroyBiTree(BiTree &T) {           //���ٶ����� 
	if (!T)
		return 0;
	else {
		if (T->lchild) 
			destroyBiTree(T->lchild);
		if (T->rchild)
			destroyBiTree(T->rchild);
	}
	free(T);
	T=NULL;
	return 1;
}


int preOrder(BiTree T) {                   //������������� 
    if(!T)
        return 0;
    visit(T->data);
    preOrder(T->lchild);
    preOrder(T->rchild);
    return 1;
} 


int inOrder(BiTree T){                   //������� 
    if(!T)
       return 0;
    inOrder(T->lchild);
    visit(T->data);
    inOrder(T->rchild);
    return 1;
}


int lastOrder(BiTree T){              //������� 
    if(!T)
       return 0;
    lastOrder(T->lchild);
    lastOrder(T->rchild);
    visit(T->data);
    return 1;
}
