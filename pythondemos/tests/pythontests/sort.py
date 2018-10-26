
lis = []
n = int(input())
for i in range(n):
    num = int(input())
    lis.append(num)
res = sorted(lis)
for i in range(n):
    print(res[i])