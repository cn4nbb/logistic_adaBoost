import random

def create_data(filename):
    data_lis = []
    for i in range(1000): #生成1000条模拟数据
        x1 = format(random.uniform(-3.0,3.0),'.2f')
        x2 = format(random.uniform(-3.0,3.0),'.2f')
        lable = str(random.randint(0,1))
        row = [x1,x2,lable]
        data_lis.append(' '.join(row))

    write_data = open(filename,'w')
    for item in data_lis:
        write_data.write(item)
        write_data.write('\n')

    write_data.close()

if __name__ == '__main__':
    create_data('./data/TestData.txt')