arr=[2,8,7,1,4]
n=len(arr)
for j in range(n-1):
    for i in range(n-j-1):
        if arr[i]>arr[i+1] :
            arr[i],arr[i+1]=arr[i+1],arr[i]
print(arr)
