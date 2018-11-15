
a = open('a.txt', 'a+')

ls = ['cool']

a.writelines(ls)

a.seek(0)   #文件指针归到文件开头

for line in a:
    print(line)