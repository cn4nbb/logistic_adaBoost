def preprocess(filename,newfilename): #缺失值赋零 处理标签
    original_data = open(filename,'r')
    new_data = open(newfilename,'w')
    lis = original_data.readlines()
    for i in range(0,len(lis)):
        row_lis = lis[i].split(' ')
        if row_lis[22] == '2' or row_lis[22] =='3': #将分类标签的值为1和2的合并为0 意为未存活
            row_lis[22] = '0'
        for j in range(0,len(row_lis)): #缺失值赋零
            if row_lis[j] == '?':
                row_lis[j] = '0'
        lis[i] = ' '.join(row_lis)

    for row in lis:
        new_data.write(row)

    new_data.close()
    original_data.close()

if __name__ == '__main__':
    #preprocess('./data/test.txt', './data/test1.txt')
    preprocess('./data/horse-colic.txt','./data/HorseTraining.txt')
    preprocess('./data/horse-colic_test.txt', './data/HorseTest.txt')

