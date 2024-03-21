# Matrix Class AVX
Programs ״simple_duble_matrix_class.cpp״ and ״simple_float_matrix_class.cpp״ are quite simple implementation of matrix class. The emphasis in these programs is placed on demonstrating methods of optimization using vector operations of the AVX2.

## Compilation:

### Clang++ 

clang++ std=c++23 -Ofast -mavx -mfma -pthread simple_duble_matrix_class.cpp -o simple_duble_matrix_class
clang++ std=c++23 -Ofast -mavx -mfma -pthread simple_float_matrix_class.cpp -o simple_float_matrix_class

### GCC 

gcc std=c++23 -O3 -mavx -mfma -pthread simple_duble_matrix_class.cpp -o simple_duble_matrix_class
gcc std=c++23 -O3 -mavx -mfma -pthread simple_float_matrix_class.cpp -o simple_float_matrix_class


