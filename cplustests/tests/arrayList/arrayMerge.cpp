#include <stdio.h>
#include <malloc.h>

#define LIST_INIT_SIZE 100
#define LISTINCREMENT 10

typedef struct list {
	int *elem;
	int length;
	int listsize;	
}arrayList,*pArrayList;

bool initList(pArrayList list);
bool mergeList(pArrayList pA,pArrayList pB,pArrayList pC);
bool insertElem(pArrayList list,int i,int e); //����Ԫ�� 
void listElems(pArrayList list); //���˳��� 

int main() {
	pArrayList pA=NULL,pB=NULL,pC=NULL;
	arrayList la,lb,lc;
	pA=&la;
	pB=&lb;
	pC=&lc;
	
	int lenA,lenB,numA,numB,i;
	
	if (initList(pA)&&initList(pB)) {
		printf("��ʼ���ɹ�");
		printf("��������Ҫ��˳���A����:\n");
		scanf("%d",&lenA);
			
		for (i=1;i<=lenA;i++) {
			scanf("%d",&numA);
			insertElem(pA,i,numA); 
		}
		
		listElems(pA);
		
		printf("��������Ҫ��˳���B����:\n");
		scanf("%d",&lenB);
			
		for (i=1;i<=lenB;i++) {
			scanf("%d",&numB);
			insertElem(pB,i,numB); 
		}
		
		listElems(pB);
		
		printf("�ϲ�AB��Ϊ:");
		mergeList(pA,pB,pC);
		
		listElems(pC);
	}
	else
		printf("��ʼ��ʧ��");
	
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


bool mergeList(pArrayList pA,pArrayList pB,pArrayList pC) {
	int *pa,*pb,*pc,*pa_last,*pb_last;
	
	pa=pA->elem; pb=pB->elem; 
	pC->listsize=pC->length=pA->length+pB->length;
	pc=pC->elem=(int *)malloc(pC->listsize*sizeof(int));
	
	if (!pC->elem)
		return false;
	
	pa_last=pA->elem+pA->length-1;
	pb_last=pB->elem+pB->length-1;
	
	while (pa<=pa_last&&pb<=pb_last) {
		if (*pa<=*pb)
			*pc++=*pa++;
		else
			*pc++=*pb++;
	}
	
	while (pa<=pa_last) {
		*pc++=*pa++;
	}
	while (pb<=pb_last) {
		*pc++=*pb++;
	}
	
	return true;
}

void listElems(pArrayList L) { //���ѭ��� 
	int i;
	for (i=0;i<L->length;i++)
		printf("%d",L->elem[i]);  //˳���Ԫ���±��0��ʼ 
}
