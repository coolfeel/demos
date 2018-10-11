#include <stdio.h>
#include <stdlib.h>

#define MaxSize 50

typedef struct SeqQueue {             //ѭ������ ����̬˳����У� 
	int data[MaxSize];
	int front,rear;
}SeqQueue,*PSeqQueue;

PSeqQueue initQueue();                //��ʼ 
bool enQueue(PSeqQueue pqueue,int x); //��� 
bool qeQueue(PSeqQueue pqueue,int *x);//���� 
bool isEmplyQueue(PSeqQueue pqueue);  //�п� 

int main() {
	
	int x;
	
	PSeqQueue pqueue=initQueue();
	
	enQueue(pqueue,2);
	
	if (enQueue(pqueue,1)) {	
		printf("��ӳɹ�\n");
	}
	
	enQueue(pqueue,3);
	
	qeQueue(pqueue,&x);
	
	return 0;
}

PSeqQueue initQueue() {
	PSeqQueue pqueue=(PSeqQueue)malloc(sizeof(SeqQueue));    //����һ�����нṹ��ռ䣬����һ��һά���� 
	if (pqueue==NULL) 
		printf("����ʧ��");
	else {
		pqueue->front=0;                                     //front��rear��ʾ�����±꣬��0��ʼ 
		pqueue->rear=pqueue->front;                          //��ʼʱ��ָ��0 
	}
	return pqueue; 
} 

bool enQueue(PSeqQueue pqueue,int x) {
	if ((pqueue->rear+1)%MaxSize==pqueue->front)             //���ʱ���� 
		return false;
	else {
		pqueue->data[pqueue->rear]=x;                        //ѭ������rearָ�����һ��Ԫ�ص���һ��λ�� 
		pqueue->rear=(pqueue->rear+1)%MaxSize;               //rear�±��1��Ҫ%MaxSize��0�� 
		return true;
	}
} 


bool qeQueue(PSeqQueue pqueue,int *x) {
	if (pqueue->front==pqueue->rear)                         //����ʱ�п� 
		return false;
	else {
		*x=pqueue->data[pqueue->front];                      //topָ��ʼ��Ԫ�أ�������ȡֵ 
		printf("����Ԫ�أ�%d  ",*x);
		pqueue->front=(pqueue->front+1)%MaxSize;             //�ٽ�front�±��1��0�� 
		return true;
	}
}



bool isEmplyQueue(PSeqQueue pqueue) {
	if (pqueue->front==pqueue->rear)
		return true;
	else
		return false;
}
