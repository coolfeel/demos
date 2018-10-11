#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

typedef struct Node {                //��ʽջ�Ľ�� 
	int data;
	struct Node *pNext;
}NODE,*PNODE;


typedef struct Stack {               //����ջ 
	PNODE pTop;                      //�ý��PNODE�洢 
	PNODE pBottom;
}STACK,*PSTACK;

void initStack(PSTACK);              //��ʼջ 
void pushStack(PSTACK,int);          //��ջ 
void traverse(PSTACK);              //������� 
bool popStack(PSTACK,int*);          //��ջ 
bool is_empty(PSTACK);              //�п� 
void clear(PSTACK);                 //ÿ��Ԫ�ض����� 

int main() {
	
	STACK S;
	int val;
	
	initStack(&S);
	pushStack(&S,1);
	pushStack(&S,2);
	popStack(&S,&val); //��Ϊֵ���ݣ������õ�ַ 
	traverse(&S);
	
	return 0;
} 


void initStack(PSTACK pS) {
	pS->pTop=(PNODE)malloc(sizeof(NODE));   //����һ������ջ�� 
	if (pS->pTop==NULL) {
		printf("����ʧ��");
		exit(-1);
	}
	else {
		pS->pBottom=pS->pTop;     //��ʼջ��ջ�� ָ��ͬһ��λ�� 
		pS->pTop->pNext=NULL;    //��pS->pBottom->pNext=NULL
	}
		
		 
}

void pushStack(PSTACK pS,int val) {
	PNODE pNew=(PNODE)malloc(sizeof(NODE));    //��ջҪ�ٴ���һ���½�� ����̬�벻���п� 
	pNew->data=val;                            
	pNew->pNext=pS->pTop;                      //��ʼtop��Ϊ��,�½�����һ��ָ��top 
	pS->pTop=pNew;								//topָ���½�㣬��ʱtopָ���½�㣨ջ������bottomָ���ʼ��㣨ջ�ף� 
	return;
}

void traverse(PSTACK pS) {
	PNODE p=pS->pTop;                          //ͷ���ָ���һ��ָ������� 
	while(p!=pS->pBottom) {                     //��ͷ��ʼ�������������β����� 
		printf("%d",p->data); 
		p=p->pNext;                            //ָ����һ�� 
	}
	return;
}

bool is_empty(PSTACK pS) {
	if (pS->pTop==pS->pBottom)               //�տ�ʼtop��bottomָ��ͬһ��λ��Ϊ�� 
		return true;
	else 
		return false;
}


bool popStack(PSTACK pS,int *pval) {
	if (is_empty(pS)) {                 //��ջ�п� 
		return false;
	}
	else {                                                 //��ʽָ������ջ��ָ�� 
		PNODE r=pS->pTop;   //����һ��ָ��r����Ȼֱ�Ӹ�ֵpS->Top=pS->Top->pNext֮���Ҳ���ԭ����pS->Top���Ͳ����ͷ����ڴ� 
		*pval=r->data;      //�Ȱ�ջ������ȡ�� 
		pS->pTop=r->pNext;  //ָ����ջ���ƶ� 
		free(r);            //�ͷ�ԭ���Ķ� 
		r=NULL;             
		return true;
	}
}

void clear(PSTACK pS) {
	if (is_empty(pS)) {           //����ǰ  �п� 
		return;
	}
	else {
		PNODE p=pS->pTop;
		PNODE q=NULL;
		while (p!=pS->pBottom) {  //pָ��,qָ������һ�� ��ÿ��Ԫ�ض����� 
			q=p->pNext;
			free(p);
			p=q;
		}                      //ȫ���ͷ�������ƶ�top=bottom 
		pS->pTop=pS->pBottom;
	}
}
