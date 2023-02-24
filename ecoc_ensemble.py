import numpy as np


def random_code_matrix(nb_class, code_len, min_hamming, min_hamming_col, min_hamming_col_complement, seed=None): 
    np.random.seed(seed) 
    W = np.zeros([nb_class, code_len]) 
    
    # generate first row 
    random_row = np.random.randint(2, size=code_len) 
    W[0] = random_row
    
    #generate codeword matrix A with condition on the hamming distance between the row respected
    W = generate_A(W, nb_class, code_len, min_hamming)
    
    # now check the conditions on the columns and swap new rows as long as the two column conditions are not respected.
    ok = False 
    while(not ok): 
        flag_column = 0
        flag_row = 0 
        row = -1

        # first check if rows respect the min hamming distance
        min_ = min_hamming_distance_row(W) 
        if(min_ < min_hamming): 
            flag_row = 1
        print("Min hamming distance Row: {}".format(min_))

        # Second check hamming distance between the complement of a column and all other columns
        min_ = min_hamming_distance_column_complement(W) 
        if(min_ < min_hamming_col_complement):
            flag_column = 1
        print("Min columns complement hamming distance: {}".format(min_))

        # fourth check hamming distance between columns 
        min_ = min_hamming_distance_column(W)
        if (min_ < min_hamming_col): 
            flag_column = 1
        print("Min hamming distance Columns: {}".format(min_))
        print('\n\n')
        
        # if flag is set to one, need to make random change 
        if flag_row == 0 and flag_column == 0 :
            ok = True
        else: 
            for i in range(1):
                index = np.random.randint(0, nb_class) 
                if flag_row == 1: 
                    index = row
                temp = True
                while(temp): 
                    random_row = np.random.randint(2, size=code_len) 
                    W[index] = random_row
                    min_ham = min_hamming_distance_row(W) 
                    if(min_ham >= min_hamming): 
                        temp = False

    print('min col: {}'.format(min_hamming_distance_column(W)))
    print('min complement col: {}'.format(min_hamming_distance_column_complement(W)))
    print('min row: {}'.format(min_hamming_distance_row(W)))
    return W




# Function to compute min hamming distance between all possible pairs of codewords from the W matrix 
def min_hamming_distance_row(W):
    '''
    Compute minimum hamming distance between rows of a matrix of codewords

            parameters: W (numpy 2d array): matrix of codewords

            Returns: min hamming_distance (int) : minimum hamming distance 
    ''' 
    nb_rows = W.shape[0]
    nb_columns = W.shape[1]
    pairs = [(i, j) for i in range(nb_rows) for j in range(i+1, nb_rows)] 
    min = W.shape[1]
    dlist = []
    for pair in pairs:
        dist = hamming_distance(W[pair[0]], W[pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist
    return min

def min_hamming_distance_row_complement(W):
    '''
    Compute minimum hamming distance between rows of a matrix of codewords

            parameters: W (numpy 2d array): matrix of codewords

            Returns: min hamming_distance (int) : minimum hamming distance 
    ''' 
    nb_rows = W.shape[0]
    nb_columns = W.shape[1]
    pairs = [(i, j) for i in range(nb_rows) for j in range(i+1, nb_rows)] 
    min = W.shape[1]
    dlist = []
    for pair in pairs:
        dist = hamming_distance(1 - W[pair[0]], W[pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist
    for pair in pairs:
        dist = hamming_distance(W[pair[0]], 1 - W[pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist
    return min

def min_hamming_distance_column(W):
    '''
    Compute minimum hamming distance betweem rows of a matrix of codeword 

            parameters: W (numpy 2d array): matrix of codewords

            Returns: min hamming_distance (int) : minimum hamming distance 
    ''' 
    nb_rows = W.shape[0]
    nb_columns = W.shape[1]

    pairs = [(i, j) for i in range(nb_columns) for j in range(i+1, nb_columns)] 
    min = W.shape[0]
    dlist = []
    for pair in pairs: 
        dist = hamming_distance(W[:, pair[0]], W[:, pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist 
    count = 0 
    tot = 0
    nb_zero = 0
    for item in dlist: 
        count+= 1 
        tot += item
        if item == 0 : 
            nb_zero += 1 

    return min 


def ham_dist_col(W, val):
    '''
    Compute minimum hamming distance betweem rows of a matrix of codeword 

            parameters: W (numpy 2d array): matrix of codewords

            Returns: min hamming_distance (int) : minimum hamming distance 
    ''' 
    flag = 0
    nb_columns = W.shape[1]
    pairs = [(i, j) for i in range(nb_columns) for j in range(i+1, nb_columns)] 
    dlist = []
    for pair in pairs: 
        dist = hamming_distance(W[:, pair[0]], W[:, pair[1]])
        print(dist)
        dlist.append(dist)
        if dist > val: 
            flag = 1
    return flag
    
def min_hamming_distance_column_complement(W):
    '''
    Compute minimum hamming distance betweem rows of a matrix of codeword 

            parameters: W (numpy 2d array): matrix of codewords

            Returns: min hamming_distance (int) : minimum hamming distance 
    ''' 
    nb_rows = W.shape[0]
    nb_columns = W.shape[1]

    pairs = [(i, j) for i in range(nb_columns) for j in range(i+1, nb_columns)] 
    min = W.shape[0]
    dlist = []
    for pair in pairs: 
        dist = hamming_distance(1 - W[:, pair[0]], W[:, pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist 
    count = 0 
    tot = 0
    for item in dlist: 
        count+= 1 
        tot += item

    for pair in pairs: 
        dist = hamming_distance(W[:, pair[0]], 1 - W[:, pair[1]])
        dlist.append(dist)
        if dist < min: 
            min = dist 
    count = 0 
    tot = 0
    for item in dlist: 
        count+= 1
        tot += item

    return min


#helper function for the method which generates a random codeword matrix
def generate_A(W, nb_class, code_len, min_hamming): 
    codes = 1 # counter for keeping track of the number of rows assigned in the matrix  
    while(codes < nb_class): 
        random_row = np.random.randint(2, size=code_len)
        W_temp = W
        W_temp[codes] = random_row
        W_temp = W_temp[0:codes+1, :] 
        hamm = min_hamming_distance_row(W_temp)
        if(hamm >= min_hamming): 
            W[codes] = random_row
            codes += 1 
    return W


# Function to compute hamming distance between two codewords
def hamming_distance(a, b): 
    '''
    Compute hamming distance between a and b 

            parameters: a, b (numpy arrays): input vectors (mush have the same length)

            Returns: hamming_distance (int) : hamming distance 
    '''
    count = 0
    if a.shape == b.shape: 
        for i, elem in enumerate(a): 
            if elem != b[i]: 
                count += 1             
    else: 
        print("Cannot compute hamming distance between two vectors of different sizes")
    return count
