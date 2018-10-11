#include <stdio.h>
#include <stdlib.h>

typedef struct Node {                   //��ʽ���� ��������� 
	int data;
	struct Node *PNode;
}Node,*PNode; 

typedef struct Queue {                  //�������нṹ���ɣ���������� 
	PNode front;
	PNode rear;
}Queue,*PQueue;

void initQueue(PQueue pqueue);         //��ʼ 
void enQueue(PQueue pqueue,int x);     //��� 
bool deQueue(PQueue pqueue,int *x);    //���� 
bool isEmptyQueue(PQueue pqueue);      //�п� 
void traverse(PQueue pqueue);          //�������� 

int main() {
	
	Queue queue;                   //���ж��� 
	PQueue pqueue=&queue;          //����ָ��,��Ȼ���� 
	int x;
	
	initQueue(pqueue);
	
	enQueue(pqueue,1);
	enQueue(pqueue,2);
	enQueue(pqueue,3);
	enQueue(pqueue,4);	
	
	deQueue(pqueue,&x);
	
	traverse(pqueue);
	
	return 0;
} 


void initQueue(PQueue pqueue) {
	pqueue->front=(PNode)malloc(sizeof(Node));     //�ȴ���һ������ͷָ�� 
	if (pqueue->front==NULL) {
		printf("����ʧ��");
		return;
	}		
	else {
		pqueue->rear=pqueue->front;               //��ʼͷ��βָ��ָ��ͬһλ�� 
		pqueue->front->PNode=NULL;                //��һ��ָ��� 
		printf("����ɹ�");
	} 	
}


void enQueue(PQueue pqueue,int x) {
	PNode pNew=(PNode)malloc(sizeof(Node));     //��ӣ������½�� 
	pNew->data=x;
	pNew->PNode=NULL;                          //�½����һ��Ϊ�� 
	pqueue->rear->PNode=pNew;                  //�½���β������ 
	pqueue->rear=pNew; 
}


bool deQueue(PQueue pqueue,int *x) {
	if (pqueue->front==pqueue->rear) {     //�����п� 
		return false;
	}
	else {
		PNode p=pqueue->front->PNode;      //��ͷ������һ����ʼ�������ӣ�ͷ��㲻��Ԫ��
		*x=p->data;                        
		pqueue->front->PNode=p->PNode;     //��ָ�뽫p�ϵ� 
		if (pqueue->rear==p) {             //������ӵ�Ϊ���һ��Ԫ�أ�ֱ�ӽ�βָ��ָ��ͷָ�� 
			pqueue->rear=pqueue->front;    //ȫ����ȥ֮��,β������գ�Ӧ����ָ���ʼ��ָ���ͷָ��һ�� 
		} 
		free(p);                           //Ȼ���ͷ�p 
		return true;
	}
}

void traverse(PQueue pqueue) {	
	PNode p=pqueue->front->PNode;               //ͷ���ָ���һ��ָ������� 
	while(p!=pqueue->rear) {                   //��ͷ��ʼ�������������β����� 
		printf("%d ",p->data); 
		p=p->PNode;                            //ָ����һ�� 
	}
	printf("%d",p->data);
}



bool isEmptyQueue(PQueue pqueue) {
	if (pqueue->front==pqueue->rear)
		return true;
	else
		return false;
}
