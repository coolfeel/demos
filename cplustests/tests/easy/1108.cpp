#include <stdio.h>

int main() {
	int m,n,t,num,temp;
	while(~scanf("%d%d",&m,&n)) {
		if (m<=0||n<=0)
			break;
		if (m<n) {
			t=m; m=n; n=t; //��֤��С�� 
		}
		
		while(1) {
			temp=m%n;     //2������������С��ȡ��
			if (temp==0)  //��Ϊ0��С���������Լ�� 
				break;
			else {
				m=n; n=temp;  //������С������������ȡ�� 
			}
		}		
		printf("%d\n",n);
	}
	return 0;
} 
