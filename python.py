#factorial
def fact(a):
    if a==1:
        return a
    return fact(a-1)*a
n=int(input("Enter a number:")) 
print(fact(n))

#prime
n=int(input("Enter a number:")) 
if n<=1:
    print("Not a prime number")
else:
    flag=0
    for i in range(2,n-1):
        if n%i==0:
            flag=1
            break
    if flag==0:
        print("Prime number")
    else:
        print("Not a prime number")

#union_intersection
def union_intersection(lst1,lst2):     
  union=list(set(lst1)|set(lst2))     
  intersection=list(set(lst1)&set(lst2))     
  return union,intersection 
nums1=[1,2,3,4,5]
nums2=[3,4,5,6,7,8]
print("Original lists:")
print(nums1)
print(nums2)
result=union_intersection(nums1,nums2)
print("\nUnion od said two lists:") 
print(result[0])
print("\nIntersection of said two lists:") 
print(result[1])

#word_occurance
def word_counter(str):
    counts=dict()
    words=str.split()
    for word in words:
        if word in counts:
            counts[word]+=1
        else:
            counts[word]=1
    return counts
print(word_counter('the quick brown fox jumps over the lazy dog'))

#multiply_matrices
def multiply_matrices(matrix1,matrix2):    
    rows1=len(matrix1)
    cols1=len(matrix1[0])
    rows2=len(matrix2)
    cols2=len(matrix2[0])
    if cols1!=rows2:
        return "Matrix multiplication not possible"     
      result=[[0 for _ in range(cols2)]
              for _ in range(rows1)]
    for i in range (rows1):
        for j in range (cols2):
            for k in range (cols1):
                result[i][j]+=matrix1[i][k]*matrix2[k][j]     
              return result
matrix1=[
    [1,2,3],
    [4,5,6]
]
matrix2=[
    [7,8],
    [9,10],
    [11,12]
]
result_matrix=multiply_matrices(matrix1,matrix2) 
if isinstance (result_matrix,str):
    print(result_matrix)
else:
    for row in result_matrix:         
      print(row)

#frequency
file=open("gfg.txt","r")
frequent_word=""
frequency=0
words=[]
for line in file:  
MIline_word=line.lower().replace(',',").replace('.',").split(" ")
    for w in line_word:
        words.append(w)
for i in range(0,len(words)):
    count=1;
    for j in range(i+1,len(words)):
        if(words[i]==words[j]):
            count=count+1
    if(count>frequency):
        frequency=count
        frequent_word=words[i]
print("Most repeated word:"+frequent_word) 
print("Frequency:"+str(frequency))
file.close()
