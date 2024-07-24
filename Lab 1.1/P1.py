def P1(l):
    out=[]
    for i in range(len(l)-1):
        for j in range(i+1,len(l)):
            if l[i]+l[j] == 10:
                a=[l[i],l[j]]
                a.sort()
                if not a in out: # to check 
                    out.append(a)
    return out

def P2(l):
    if len(l)<3:
        return "Range determination not possible"
    max, min=max_min(l)
    return max-min
    

def max_min(l):
    max=l[0]
    min=l[0]
    for i in l:
        if i>max:
            max=i
        if i<min:
            min=i
    return max,min

def P3(l,original,n):
    
    output = [[0 for j in range(n)] for i in range(n)] 
    
    print(original)
    for r in range(0,n):
        for c in range(0,n):
            s=0
            for p in range(0,n):
                s+=l[r][p]*original[p][c]
            output[r][c]=s
    l=output 
    print(l)  
    return output
                    
def P4(l):
    a={}
    for i in l:
        if i in a:
            a[i]+=1
        else:
            a[i]=1
    values=list(a.values())
    max=values[0]
    for i in a.values():
        if i>max:
            max=i
    
    output=[]

    for i in a.keys():
        if a[i]==max:
            output.append(i)
    return output,max

        

            
    



if __name__ == "__main__":
    x = input("Enter a choice from 1-4: ") # Switch case to select the program to execute
    match x:
        case "1":
            output = P1([2,7,4,1,3,6]) # to count pairs of elements with sum equal to 10
            if len(output) == 0:
                print("No pairs found")
            else:
                print("All pairs which have sum 10 are:-"+str(output))
        case "2":
            output = P2([1,2,3,4,8,2,6,9,7])
            if type(output) == str:
                print(output)
            else:
                print("Range is:- "+str(output))

        case "3":
            n=int(input("Enter the number of rows for the square matrix: "))
            l=[[int(input("Enter element for row:"+str(i+1)+" column:"+str(j+1)+"\n")) for j in range (n)] for i in range(n)]
            
            original=l # to store the output matrix
            # for i in range(n):
            #     a=[]
            #     for j in range(n):
            #         a.append(l[i][j])
            #         original.append(a)  
            
            for m in range(1,n):
                l = P3(l,original,n)
            print("The matrix power A^n is:-\n")
            for i in range(n):
                for j in range(n):
                    print(l[i][j],end=" ")
                print()
        case "4":
            l=input("Enter a string: ")
            output,max = P4(l)
            print("The most frequent character(s) is/are:- "+str(output)+"\nFrequency:- "+str(max))

        case _:
            print("Invalid choice")