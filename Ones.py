def Ones(mat,R,C): 
  
    perimeter = 0; 
    for i in range(0, R): 
        for j in range(0, C): 
            if (mat[i][j]==0): 
                perimeter += (1); 
  
    return perimeter; 
