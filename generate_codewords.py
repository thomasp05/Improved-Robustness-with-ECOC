import numpy as np 
import ecoc_ensemble
import sys

if __name__ == "__main__": 
    # load fileName for the .pth model from the terminal (the user must pass it when calling the script from the terminal)
    arg_list = sys.argv
    nb_classes = int(arg_list[1]) 
    code_len = int(arg_list[2])
    min_hamming_rows = int(arg_list[3])
    min_hamming_cols = int(arg_list[4])
    min_hamming_cols_complement = int(arg_list[5])

    codeword_matrix = ecoc_ensemble.random_code_matrix(nb_classes, code_len, min_hamming_rows, min_hamming_cols, min_hamming_cols_complement) 

    print(codeword_matrix)
    np.save('codeword_' + '_'+ str(nb_classes) + '_'+ str(code_len) + '_'+ str(min_hamming_rows) + '_'+ str(min_hamming_cols) + '.npy', codeword_matrix, allow_pickle=True)
    