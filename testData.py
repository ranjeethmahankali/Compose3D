from ops import *
dtSet = dataset('dataset/')

num = 10
batch = dtSet.next_batch(247)
batch = dtSet.next_batch(num)
for i in range(num):
    img = toImage(batch[0][i: i+1])
    arr = batch[1][1]
    print(len(arr))
    img.save('dataset/%s.png'%i)