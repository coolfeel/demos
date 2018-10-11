#include <stdio.h>
#include <stdlib.h>

#define MaxSize 50

typedef struct SeqStack {    //˳��ջ 
	int data[MaxSize];
	int top;               //ջ��ָ��  top���������±�  ��0��ʼ 
}SeqStack,*PSeqStack;


PSeqStack initSeqStack();             //��ʼջ 
bool isEmptySatck(SeqStack stack);   //�ж��Ƿ�Ϊ�� 
bool pushStack(PSeqStack pstack,int x); //��ջ 
int getTop(PSeqStack pstack);            //�õ�ջ��Ԫ�� 
bool popStack(PSeqStack pstack);         //��ջ


int main() {
	
	PSeqStack pstack=initSeqStack();
	
	if (pstack) {
		pushStack(pstack,1);
		pushStack(pstack,2);
		pushStack(pstack,3);
		printf("%d",getTop(pstack));
	}
	
	for (int i=0;i<3;i++) {
		popStack(pstack);
	}
	
	return 0;
} 

PSeqStack initSeqStack() {
	PSeqStack pstack=(PSeqStack)malloc(sizeof(SeqStack));    //ֱ�ӷ���һ���ṹ�� 
	if (pstack==NULL)
		printf("����ʧ��");
	else
		pstack->top=-1;      //ջ�գ�ջ��Ϊ-1������1��Ԫ��ָ��0��topֱ��ָ��Ԫ�� 
	return pstack;                                       
}


bool isEmptySatck(SeqStack stack) {
	if (stack.top==-1)           
		return true;
	else
		return false;
}


bool pushStack(PSeqStack pstack,int x) {
	if (pstack->top==MaxSize-1) {               //����ʱ���� 
		printf("ջ��");
		return false;
	}
	else {
		pstack->top++;
		pstack->data[pstack->top]=x;
		return true;
	}			
}


bool popStack(PSeqStack pstack) {              //��ջʱ�п� 
	if (pstack->top==-1) {
		printf("ջ��");
		return false;
	}		
	else {
		printf ("ջ��Ԫ�أ�%d",pstack->data[pstack->top]);
		pstack->top--;
	}
		
}

int getTop(PSeqStack pstack) {               //ֱ�ӵõ���Ԫ�� 
	return pstack->data[pstack->top];
}
