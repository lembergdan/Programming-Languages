#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <thread>


using namespace std;

long Get_Time() {
    using chrono::high_resolution_clock;
    auto t = high_resolution_clock::now();
    auto nanosec = t.time_since_epoch();
    return nanosec.count() / 1000000;
}

class Matrix{
public:
    double *p;
    int rows;
    int cols;
    
    Matrix() : p(nullptr), rows(0), cols(0) {}
    Matrix(int rows_, int cols_) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) {}
    Matrix(int rows_, int cols_, double val) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) {
        for(int i = 0; i < rows * cols; i++)
            p[i] = val;
    }
    
    Matrix(int rows_, int cols_, double a, double b) : p(new double[rows_ * cols_]), rows(rows_), cols(cols_) { // Random matrix a(i,j) ~ U(a, b)
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(a, b);
        
        for(int i = 0; i < rows * cols; i++) p[i] = dis(gen);
    }
    
    Matrix(string path) {
        ifstream t(path);
        string str((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());
            
        auto in_float = [](char ch) { return ('0' <= ch && ch <= '9') || (ch == '.'); };
        int rows_ = 0, cols_ = 0;
        
        for (int i = 0; i < str.size(); i++)
            if(str[i] == '\n') rows_++;
        
        for (int i1 = 0, i2 = 0; i2 < str.size() && str[i2] != '\n'; ) {
            for(i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for(i2 = i1;  in_float(str[i2]) && i2 < str.size(); i2++) {}
            if(i1 != i2) cols_++;
        }
        
        rows = rows_; cols = cols_;
        p = new double[rows * cols];
        
        for (int i1 = 0, i2 = 0, j = 0; i2 < str.size(); ) {
            for(i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for(i2 = i1;  in_float(str[i2]) && i2 < str.size(); i2++) {}
            if(i1 != i2) p[j++] = stof(str.substr(i1, i2 - i1));
        }
    }
    
    Matrix(Matrix& m) : p(new double[m.rows * m.cols]), rows(m.rows), cols(m.cols) {
        for(int i = 0; i < rows * cols; i++)
            p[i] = m.p[i];
    }
    
    Matrix(Matrix&& m) : rows(m.rows), cols(m.cols) {
        p = m.p;
        m.p = nullptr;
    }
    
    friend bool eq(Matrix&  a, Matrix&  b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix&  b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&  a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    
    Matrix& operator = (Matrix& m) {
        if(p == m.p) return *this;
        if(eq(*this, m)) {
            for(int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        else {
            delete[] p;
            rows = m.rows;
            cols = m.cols;
            p = new double[rows * cols];
            for(int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        return *this;
    }
    
    Matrix& operator = (Matrix&& m) {
        if(p == m.p) return *this;
        p = m.p;
        m.p = nullptr;
        return *this;
    }
    
    ~Matrix(){
        if(p) delete[] p;
    }
    
    double& operator [] (int i, int j) {
        if(0 <= i && i < rows && 0 <= j && j < cols)
            return p[i * cols + j];
        cerr << "Error of index in operator []." << endl;
        return p[0];
    }
    
    /////////////////////////////// transpose ///////////////////////////////
    Matrix t() {
        Matrix tr(cols, rows);
        int r = (rows / 4) * 4 - 4;
        int c = (cols / 4) * 4 - 4;
        
        for (int i = 0; i < rows - 4; i+=4)
            for (int j = 0; j < cols - 4; j+=4)
                Tools::transpose4x4(p + i * cols + j, tr.p + j * rows + i, cols, rows);
        
        for (int i = r; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tr[j, i] = p[i * cols + j];
        
        for (int i = 0; i < rows; i++)
            for (int j = c; j < cols; j++)
                tr[j, i] = p[i * cols + j];
            
        return tr;
    }
    
    /////////////////////////////////// + ///////////////////////////////////
    friend Matrix operator + (Matrix& a, Matrix& b) {
        if(!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }
        
        Matrix s(a.rows, a.cols);
        for(int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] + b.p[i];
        return s;
    }
    
    friend Matrix operator + (Matrix&& a, Matrix&& b) {
        if(!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }
        for(int i = 0; i < b.rows * b.cols; i++)
            b.p[i] += a.p[i];
        return std::move(b);
    }
    
    friend Matrix operator + (Matrix&& a, Matrix& b) { return std::move(b) + std::move(a); }
    friend Matrix operator + (Matrix& a, Matrix&& b) { return std::move(a) + std::move(b); }
    
    Matrix& operator += (Matrix&& m) {
        if(!eq(*this, m)) {
            cerr << "Error of matrix size in operator +=." << endl;
            return *this;
        }
        for(int i = 0; i < rows * cols; i++)
            p[i] += m.p[i];
        return *this;
    }
    
    Matrix& operator += (Matrix& m) { return operator+=(std::move(m)); }
    /////////////////////////////////// - ///////////////////////////////////
    friend Matrix operator - (Matrix& a, Matrix& b) {
        if(!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        Matrix s(a.rows, a.cols);
        for(int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] - b.p[i];
        return s;
    }
    
    friend Matrix operator - (Matrix&& a, Matrix&& b) {
        if(!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for(int i = 0; i < a.rows * a.cols; i++)
            a.p[i] -= b.p[i];
        return std::move(a);
    }
    
    friend Matrix operator - (Matrix&& a, Matrix& b) { return std::move(a) - std::move(b); }
    
    friend Matrix operator - (Matrix& a, Matrix&& b) {
        if(!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for(int i = 0; i < a.rows * a.cols; i++)
            b.p[i] = a.p[i] - b.p[i];
        return std::move(b);
    }
    
    Matrix& operator -= (Matrix&& m) {
        if(!eq(*this, m)) {
            cerr << "Error of matrix size in operator -=." << endl;
            return *this;
        }
        for(int i = 0; i < rows * cols; i++)
            p[i] -= m.p[i];
        return *this;
    }
    
    Matrix& operator -= (Matrix& m) { return operator-=(std::move(m)); }
    
    /////////////////////////////////// * ///////////////////////////////////
    friend Matrix operator * (Matrix& a, double b) {
        Matrix prod(a.rows, a.cols);
        for(int i = 0; i < a.rows * a.cols; i++)
            prod.p[i] = a.p[i] * b;
        return prod;
    }
    
    friend Matrix operator * (Matrix&& a, double b) {
        for(int i = 0; i < a.rows * a.cols; i++)
             a.p[i] *= b;
        return std::move(a);
    }
    
    Matrix& operator *= (Matrix&& m) {
        if(!eq(*this, m)) {
            cerr << "Error of matrix size in operator *." << endl;
            return *this;
        }
        for(int i = 0; i < rows * cols; i++)
            p[i] *= m.p[i];
        return *this;
    }
    
    Matrix& operator *= (Matrix& m) { return operator*=(std::move(m)); }
    
    friend Matrix operator * (double b, Matrix&  a) { return a * b; }
    friend Matrix operator * (double b, Matrix&& a) { return a * b; }
    
    friend Matrix operator * (Matrix&& a, Matrix&& b) {
        if (a.cols != b.rows) {
            cerr << "Error of matrix size in operator *." << endl;
            return Matrix();
        }
        Matrix ret;
        ret.p = Tools::mult_thread_padd(a.rows, a.p, b.p, a.cols, b.cols, b.cols, Tools::dim_th, Tools::n_th);
        ret.rows = a.rows;
        ret.cols = b.cols;
        return ret;
    }
    
    friend Matrix operator * (Matrix&  a, Matrix&  b) { return std::move(a) * std::move(b); }
    friend Matrix operator * (Matrix&& a, Matrix&  b) { return std::move(a) * std::move(b); }
    friend Matrix operator * (Matrix&  a, Matrix&& b) { return std::move(a) * std::move(b); }
    
    friend ostream& operator << (ostream& out, Matrix&& m) {
        for(int i = 0; i < m.rows - 1; i++) {
            for(int j = 0; j < m.cols; j++)
                out << m[i, j] << "\t";
            out << endl;
        }
        for(int j = 0; j < m.cols; j++)
            out << m[m.rows - 1, j] << "\t";
        return out;
    }
    
    friend ostream& operator << (ostream& out, Matrix& m) {
        return out << std::move(m);
    }

    /////////////////////////////////// Matrix Operation Tools Class ///////////////////////////////////
     struct Tools {
         static const int L1 = 32 * 1024;  // Size of L1 cache
         static const int L2 = 256 * 1024; // Size of L2 cache
         static const int n_th = 8;        // Number of threads
         static const int dim_th = 0;      // Dimension for cutting (to multithreading)
         
         // Transposition of 4 __m256d registers
         inline static void transpose_4_rows_pd(__m256d &row0, __m256d &row1, __m256d &row2, __m256d &row3)
         {
            __m256d &&t0 = _mm256_shuffle_pd(row0, row1, 0x0);
            __m256d &&t2 = _mm256_shuffle_pd(row0, row1, 0xF);
            __m256d &&t1 = _mm256_shuffle_pd(row2, row3, 0x0);
            __m256d &&t3 = _mm256_shuffle_pd(row2, row3, 0xF);
                                                                                 
            row0 = _mm256_permute2f128_pd(t0, t1, 0x20);
            row1 = _mm256_permute2f128_pd(t2, t3, 0x20);
            row2 = _mm256_permute2f128_pd(t0, t1, 0x31);
            row3 = _mm256_permute2f128_pd(t2, t3, 0x31);
         }
         
         // Transposition of 4x4 matrix
         inline static void transpose4x4(double *&&A, double *&&B, const int lda, const int ldb) {
             __m256d &&row0 = _mm256_loadu_pd(A + 0 * lda);
             __m256d &&row1 = _mm256_loadu_pd(A + 1 * lda);
             __m256d &&row2 = _mm256_loadu_pd(A + 2 * lda);
             __m256d &&row3 = _mm256_loadu_pd(A + 3 * lda);
             
             transpose_4_rows_pd(row0, row1, row2, row3);
             
             _mm256_storeu_pd(B + 0 * ldb, row0);
             _mm256_storeu_pd(B + 1 * ldb, row1);
             _mm256_storeu_pd(B + 2 * ldb, row2);
             _mm256_storeu_pd(B + 3 * ldb, row3);
         }
         
        // Zero initialization of the block (m x n) in the matrix ("c" - start of the block, ldc - namber of colums in the matrix)
        static void init_c(int m, int n, double *c, int ldc)
        {
            for (int i = 0; i < m; i++, c += ldc)
                for (int j = 0; j < n; j += 4)
                    _mm256_storeu_pd(c + j, _mm256_setzero_pd());
        }

        // Multiplication of (6 x k) block of "a" and (k x 8) block of "b" ("b" - reordered) and streing it to (6 x 8) block in "c"
        static void kernel(int k, const double *a, const double *b, double *c, int lda, int ldb, int ldc)
        {
            __m256d a0, a1, b0, b1;
        
            __m256d &&c00 = _mm256_setzero_pd();    __m256d &&c01 = _mm256_setzero_pd();
            __m256d &&c10 = _mm256_setzero_pd();    __m256d &&c11 = _mm256_setzero_pd();
            __m256d &&c20 = _mm256_setzero_pd();    __m256d &&c21 = _mm256_setzero_pd();
            __m256d &&c30 = _mm256_setzero_pd();    __m256d &&c31 = _mm256_setzero_pd();
            __m256d &&c40 = _mm256_setzero_pd();    __m256d &&c41 = _mm256_setzero_pd();
            __m256d &&c50 = _mm256_setzero_pd();    __m256d &&c51 = _mm256_setzero_pd();

            const int offset0 = lda * 0;         const int offset3 = lda * 3;
            const int offset1 = lda * 1;         const int offset4 = lda * 4;
            const int offset2 = lda * 2;         const int offset5 = lda * 5;

            for (int i = 0; i < k; i++)
            {
                b0 = _mm256_loadu_pd(b + 0);                  b1 = _mm256_loadu_pd(b + 4);

                a0 = _mm256_broadcast_sd(a + offset0);        a1 = _mm256_broadcast_sd(a + offset1);

                c00 = _mm256_fmadd_pd(a0, b0, c00);           c10 = _mm256_fmadd_pd(a1, b0, c10);
                c01 = _mm256_fmadd_pd(a0, b1, c01);           c11 = _mm256_fmadd_pd(a1, b1, c11);

                a0 = _mm256_broadcast_sd(a + offset2);        a1 = _mm256_broadcast_sd(a + offset3);

                c20 = _mm256_fmadd_pd(a0, b0, c20);           c30 = _mm256_fmadd_pd(a1, b0, c30);
                c21 = _mm256_fmadd_pd(a0, b1, c21);           c31 = _mm256_fmadd_pd(a1, b1, c31);

                a0 = _mm256_broadcast_sd(a + offset4);        a1 = _mm256_broadcast_sd(a + offset5);

                c40 = _mm256_fmadd_pd(a0, b0, c40);           c50 = _mm256_fmadd_pd(a1, b0, c50);
                c41 = _mm256_fmadd_pd(a0, b1, c41);           c51 = _mm256_fmadd_pd(a1, b1, c51);

                b += ldb; a++;
            }
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c00, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c01, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c10, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c11, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c20, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c21, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c30, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c31, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c40, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c41, _mm256_loadu_pd(c + 4)));
            c += ldc;
            _mm256_storeu_pd(c + 0, _mm256_add_pd(c50, _mm256_loadu_pd(c + 0)));
            _mm256_storeu_pd(c + 4, _mm256_add_pd(c51, _mm256_loadu_pd(c + 4)));
        }

        // Reordering of (k x 16) block of B
        static void reorder_b(int k, const double *b, int ldb, double *b_tmp)
        {
            for (int i = 0; i < k; i++, b += ldb, b_tmp += 8)
            {
                _mm256_storeu_pd(b_tmp + 0, _mm256_loadu_pd(b + 0));
                _mm256_storeu_pd(b_tmp + 4, _mm256_loadu_pd(b + 4));
            }
        }

         // Reordering of (m x k) block of A
         static void reorder_a(const double* a, int m, int k, int lda, double* a_tmp){
             for(int i = 0; i < m; i++)
                 for(int j = 0; j < k; j++)
                     a_tmp[i * k + j] = a[i * lda + j];
         }
         
         //Localization
         static void macro(int m, int k, int n, const double* a, const double* b, double* c, double* b_tmp, int lda, int ldb, int ldc)
         {
             for (int j = 0; j < n; j += 8)
             {
                 reorder_b(k, b + j, ldb, b_tmp);
                 for (int i = 0; i < m; i += 6)
                     kernel(k, a + i * lda, b_tmp, c + i * ldc + j, lda, 8, ldc);
             }
         }
         
         // Product of matrices A (m x k) and B (k x n)
         static void mult(int m, int k, int n, const double* a, const double* b, double* c, int lda, int ldb, int ldc)
         {
             int k_block = std::min(L1 / 8 / 8, k);
             int m_block = std::min(L2 / 8 / k_block, m) / 6 * 6;
             double* b_tmp = new double[k_block * 8];
             double* a_tmp = new double[k_block * m_block];
             
             for(int t = 0; t < k; t += k_block)
             {
                 int dk = std::min(k, t + k_block) - t;
                 for (int i = 0; i < m; i += m_block)
                 {
                     int dm = std::min(m, i + m_block) - i;
                     if(t == 0)
                         init_c(dm, n, c + i * ldc, ldc);

                     reorder_a(a + i * lda + t, dm, dk, lda, a_tmp);
                     macro(dm, dk, n, a_tmp, b + t * ldb, c + i * ldc, b_tmp, dk, ldb, ldc);
                 }
             }

             delete[] b_tmp;
             delete[] a_tmp;
         }

    // Multithreaded product of matrices A (m x k) and B (k x n)
        static double* mult_thread(int m, const double *a, const double *b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int m_t;
            thread t[n_thread];
            double *c = new double[m * ldc];
        
            switch (dim_thread) {
                case 0:
                    m_t = m / n_thread;
                    for(int i = 0; i < n_thread; i++)
                        t[i] = thread( [&, i](){ mult(m_t, lda, ldc, a + i * m_t * lda, b, c + i * m_t * ldc, lda, ldb, ldc); } );
                    break;
                case 1:
                    m_t = ldc / n_thread;
                    for(int i = 0; i < n_thread; i++)
                        t[i] = thread( [&, i](){ mult(m, lda, m_t, a, b + i * m_t, c + i * m_t, lda, ldb, ldc); } );
                    break;
                default:
                    cerr << "Error in parametr 'dim_thread' in function 'mult_thread'." << endl;
                    return nullptr;
            }
            for(int i = 0; i < n_thread; i++)
                t[i].join();
        
            return c;
        }

        static double* padd_mat(const double* a, int m, int n, int new_m, int new_n) {
            double* p = new double[new_m * new_n];
            int t = 0;
        
            for(int i = 0, j; i < m; i++) {
                for(j = 0; j < n; j++)
                    p[t++] = a[i * n + j];
                for(; j < new_n; j++)
                    p[t++] = 0;
            }
        
            for(; t < new_m * new_n; t++)
                p[t] = 0;
        
            return p;
        }

        static double* unpadd_mat(const double* a, int m, int n, int new_m, int new_n) {
            double* p = new double[new_m * new_n];
        
            for(int i = 0, j = 0, t = 0; i < new_m; i++, j += (n - new_n))
                for(int k = 0; k < new_n; k++, j++, t++)
                    p[t] = a[j];
            
            return p;
        }

        static double* mult_thread_padd(int m, const double *a, const double *b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int c, m_new, lda_new, ldb_new, ldc_new;
        
            switch (dim_thread) {
                case 0:
                    c = 6 * n_thread;
                    lda_new = (lda % 8 == 0)  ? lda : (lda / 8) * 8 + 8;
                    ldb_new = (ldb % 8  == 0) ? ldb : (ldb / 8) * 8 + 8;
                    ldc_new = (ldc % 8  == 0) ? ldc : (ldc / 8) * 8 + 8;
                    m_new   = (m   % c  == 0)  ? m   : (m   / c)  * c  + c;
                    break;
                case 1:
                    c = 8 * n_thread;
                    lda_new = (lda % 8 == 0) ? lda : (lda / 8) * 8 + 8;
                    ldb_new = (ldb % c  == 0) ? ldb : (ldb / c) * c + c;
                    ldc_new = (ldc % c  == 0) ? ldc : (ldc / c) * c + c;
                    m_new   = (m   % 6  == 0) ? m   : (m   / 6) * 6  + 6;
                    break;
                default:
                    cerr << "Error in parametr 'dim_thread' in function 'mult_thread_padd'." << endl;
                    return nullptr;
            }
        
            double *a_padd = nullptr, *b_padd = nullptr, *c_padd = nullptr, *ret = nullptr;
            bool is_a_padd = m_new   != m   || lda_new != lda;
            bool is_b_padd = lda_new != lda || ldb_new != ldb;
        
            if(is_a_padd) a_padd = padd_mat(a, m, lda, m_new, lda_new);
            if(is_b_padd) b_padd = padd_mat(b, lda, ldb, lda_new, ldb_new);

            if(is_a_padd && is_b_padd) {
            
                c_padd = mult_thread(m_new, a_padd, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] b_padd;
                delete[] c_padd;
            }
            if(is_a_padd && !is_b_padd) {
                c_padd = mult_thread(m_new, a_padd, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);
             
                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] c_padd;
            }
            if(!is_a_padd && is_b_padd) {
                c_padd = mult_thread(m_new, a, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);
            
                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] b_padd;
                delete[] c_padd;
            }
            if(!is_a_padd && !is_b_padd) {
                ret = mult_thread(m_new, a, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);
            }
            return ret;
        }
    };
};


double* mult(int M, int K, int N, double *A, double *B) {

    double *C = new double[M * N];
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            C[i * N + j] = 0;
    
    for(int i = 0; i < M; i++)
        for(int k = 0; k < K; k++)
            for(int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    return C;
    
}

int main(int argc, const char * argv[]) {
    long t1, t2;
    int m = 4021;
    int n = 4031;
    int k = 4073;

    Matrix a(m, k, 0, 1);
    Matrix b(k, n, 0, 1);

    t1 = Get_Time();
    Matrix ab = a * b + (b.t() * a.t()).t();
    t2 = Get_Time();

    cout << "ab time = " << t2 - t1 << endl;
    cout << "b[40, 120] = " << ab[40, 120] << endl;
    
    return 0;
}
