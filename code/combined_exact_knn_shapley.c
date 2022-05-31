#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <alloca.h>

#include "tsc_x86.h"
#include "benchmark.h"
#include "knn_exact.h"
#include "combined_exact_knn_shapley.h"
#include "base_exact_shapley.h"


extern double* dist_gt;
extern double* dist_gt_row;

// ######################################################
// START Quicksort implementation taken form glibc v2.35
// Minor tweaks, now sorts two arrays based on input
// The input are two concatted arrays A||B
// During sorting, every swap in A is also applied (MIMICED) to B
// ######################################################

/* Byte-wise swap two items of size SIZE. */
#define SWAP(a, b, size)                \
  do									\
    {								    \
      size_t __size = (size);	        \
      char *__a = (a), *__b = (b);      \
      do							    \
	{		         		            \
    char __tmp = *__a;				    \
	  *__a++ = *__b;				    \
	  *__b++ = __tmp;			        \
	} while (--__size > 0);	     	    \
    } while (0)

/* Byte-wise swap two items of size SIZE. */
#define SWAPMIMIC(a, b, total_elems, size)                \
  do									\
    {								    \
      size_t __size = (size);	        \
      size_t __offset = (size) * (total_elems);\
      char *__a = (a)+__offset, *__b = (b)+__offset;      \
      do							    \
	{		         		            \
    char __tmp = *__a;				    \
	  *__a++ = *__b;				    \
	  *__b++ = __tmp;			        \
	} while (--__size > 0);	     	    \
    } while (0)

/* Discontinue quicksort algorithm when partition gets below this size.
   This particular magic number was chosen to work best on a Sun 4/260. */
#define MAX_THRESH 4

/* Stack node declarations used to store unfulfilled partition obligations. */
typedef struct
  {
    char *lo;
    char *hi;
  } stack_node;

/* The next 4 #defines implement a very fast in-line stack abstraction. */
/* The stack needs log (total_elements) entries (we could even subtract
   log(MAX_THRESH)).  Since total_elements has type size_t, we get as
   upper bound for log (total_elements):
   bits per byte (CHAR_BIT) * sizeof(size_t).  */
#define STACK_SIZE	(CHAR_BIT * sizeof (size_t))
#define PUSH(low, high)	((void) ((top->lo = (low)), (top->hi = (high)), ++top))
#define	POP(low, high)	((void) (--top, (low = top->lo), (high = top->hi)))
#define	STACK_NOT_EMPTY	(stack < top)


/* Order size using quicksort.  This implementation incorporates
   four optimizations discussed in Sedgewick:

   1. Non-recursive, using an explicit stack of pointer that store the
      next array partition to sort.  To save time, this maximum amount
      of space required to store an array of SIZE_MAX is allocated on the
      stack.  Assuming a 32-bit (64 bit) integer for size_t, this needs
      only 32 * sizeof(stack_node) == 256 bytes (for 64 bit: 1024 bytes).
      Pretty cheap, actually.

   2. Chose the pivot element using a median-of-three decision tree.
      This reduces the probability of selecting a bad pivot value and
      eliminates certain extraneous comparisons.

   3. Only quicksorts TOTAL_ELEMS / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.

   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (total_elems)
      stack size is needed (actually O(1) in this case)!  */

void
quicksort (void *const pbase, size_t total_elems, size_t size)
{
  char *base_ptr = (char *) pbase;

  const size_t max_thresh = MAX_THRESH * size;

  if (total_elems == 0)
    /* Avoid lossage with unsigned arithmetic below.  */
    return;

  if (total_elems > MAX_THRESH)
    {
      char *lo = base_ptr;
      char *hi = &lo[size * (total_elems - 1)];
      stack_node stack[STACK_SIZE];
      stack_node *top = stack;

      PUSH (NULL, NULL);

      while (STACK_NOT_EMPTY)
        {
          char *left_ptr;
          char *right_ptr;

	  /* Select median value from among LO, MID, and HI. Rearrange
	     LO and HI so the three values are sorted. This lowers the
	     probability of picking a pathological pivot value and
	     skips a comparison for both the LEFT_PTR and RIGHT_PTR in
	     the while loops. */

	  char *mid = lo + size * ((hi - lo) / size >> 1);

	  if (*(double*)mid < *(double*)lo) {
        SWAP (mid, lo, size);
	    SWAPMIMIC (mid, lo, total_elems, size);
      }
	  if (*(double *)hi < *(double *)mid) {
	    SWAP (mid, hi, size);
	    SWAPMIMIC (mid, hi, total_elems, size);
      }
	  else
	    goto jump_over;
	  if (*(double *)mid < *(double *)lo) {
	    SWAP (mid, lo, size);
	    SWAPMIMIC (mid, lo, total_elems, size);
        SWAP (mid, lo, size);
        SWAPMIMIC (mid, lo, total_elems, size);
      }
	jump_over:;

	  left_ptr  = lo + size;
	  right_ptr = hi - size;

	  /* Here's the famous ``collapse the walls'' section of quicksort.
	     Gotta like those tight inner loops!  They are the main reason
	     that this algorithm runs much faster than others. */
	  do
	    {
	    //   while (MAC_COMPAR_BLOCK( left_ptr, mid) < 0)
	      while (*(double *)left_ptr< *(double *)mid)
		left_ptr += size;

	      while (*(double *)mid< *(double *)right_ptr)
	    //   while (MAC_COMPAR_BLOCK(mid, right_ptr) < 0)
		right_ptr -= size;

	      if (left_ptr < right_ptr) {
            SWAP (left_ptr, right_ptr, size);
            SWAPMIMIC (left_ptr, right_ptr, total_elems, size);
            if (mid == left_ptr)
                mid = right_ptr;
            else if (mid == right_ptr)
                mid = left_ptr;
            left_ptr += size;
            right_ptr -= size;
		}
	      else if (left_ptr == right_ptr)
		{
		  left_ptr += size;
		  right_ptr -= size;
		  break;
		}
	    }
	  while (left_ptr <= right_ptr);

          /* Set up pointers for next iteration.  First determine whether
             left and right partitions are below the threshold size.  If so,
             ignore one or both.  Otherwise, push the larger partition's
             bounds on the stack and continue sorting the smaller one. */

          if ((size_t) (right_ptr - lo) <= max_thresh)
            {
              if ((size_t) (hi - left_ptr) <= max_thresh)
		/* Ignore both small partitions. */
                POP (lo, hi);
              else
		/* Ignore small left partition. */
                lo = left_ptr;
            }
          else if ((size_t) (hi - left_ptr) <= max_thresh)
	    /* Ignore small right partition. */
            hi = right_ptr;
          else if ((right_ptr - lo) > (hi - left_ptr))
            {
	      /* Push larger left partition indices. */
              PUSH (lo, right_ptr);
              lo = left_ptr;
            }
          else
            {
	      /* Push larger right partition indices. */
              PUSH (left_ptr, hi);
              hi = right_ptr;
            }
        }
    }

  /* Once the BASE_PTR array is partially sorted by quicksort the rest
     is completely sorted using insertion sort, since this is efficient
     for partitions below MAX_THRESH size. BASE_PTR points to the beginning
     of the array to sort, and END_PTR points at the very last element in
     the array (*not* one beyond it!). */

#define min(x, y) ((x) < (y) ? (x) : (y))

  {
    char *const end_ptr = &base_ptr[size * (total_elems - 1)];
    char *tmp_ptr = base_ptr;
    char *thresh = min(end_ptr, base_ptr + max_thresh);
    char *run_ptr;

    /* Find smallest element in first threshold and place it at the
       array's beginning.  This is the smallest array element,
       and the operation speeds up insertion sort's inner loop. */

    for (run_ptr = tmp_ptr + size; run_ptr <= thresh; run_ptr += size)
    //   if (MAC_COMPAR_BLOCK(run_ptr, tmp_ptr) < 0)
	  if (*(double *)run_ptr < *(double *)tmp_ptr)
        tmp_ptr = run_ptr;

    if (tmp_ptr != base_ptr) {
        SWAP (tmp_ptr, base_ptr, size);
        SWAPMIMIC (tmp_ptr, base_ptr, total_elems, size);
    }

    /* Insertion sort, running from left-hand-side up to right-hand-side.  */
    /* Implementation heavily inspired by https://www.geeksforgeeks.org/insertion-sort/ */


    double* arr = (double*)base_ptr;
    size_t n = total_elems;
    int i, j;
    double key;
    double keyMIMIC;
    for (i = 1; i < n; i++) {
        key = arr[i];
        keyMIMIC = arr[i+total_elems];

        j = i - 1;
 
        /* Move elements of arr[0..i-1], that are
          greater than key, to one position ahead
          of their current position */
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            arr[j + total_elems + 1] = arr[j + total_elems];
            j = j - 1;
        }
        arr[j + 1] = key;
        arr[j + total_elems + 1] = keyMIMIC;
    }

    // run_ptr = base_ptr + size;
    // while ((run_ptr += size) <= end_ptr) {
    //     tmp_ptr = run_ptr - size;
    //     // while (MAC_COMPAR_BLOCK(run_ptr, tmp_ptr) < 0)
    //     while (*(double *)run_ptr < *(double *)tmp_ptr)
    //         tmp_ptr -= size;

    //     tmp_ptr += size;
    //     if (tmp_ptr != run_ptr) {
    //         char *trav;
    //         char *travMIMIC;

    //         trav = run_ptr + size;
    //         travMIMIC = trav + total_elems * size;
    //         while (--trav >= run_ptr) {
    //             travMIMIC = trav + total_elems * size;
    //             char c = *trav;
    //             char cMIMIC = *travMIMIC;
    //             char *hi, *lo;
    //             char *hiMIMIC, *loMIMIC;

    //             for (hi = lo = trav; (lo -= size) >= tmp_ptr; hi = lo){
    //                 hiMIMIC = hi + total_elems * size;
    //                 loMIMIC = lo + total_elems * size;
    //                 *hi = *lo;
    //                 printf("insert hi: %f\n", *(double*)hi);
    //                 *hiMIMIC = *loMIMIC;
    //             }
    //             *hi = c;
    //             printf("insert hic: %f\n", *(double*)hi);
    //             *hiMIMIC = cMIMIC;
    //             printf("insert hiMIMIC: %f\n", *(double*)hiMIMIC);
    //         }
    //     }
    //   }
  }
}

// ######################################################
// END Quicksort implementation taken form glibc v2.35
// ######################################################


static inline
double vec_sum(__m256d vec) {
    __m128d vlow  = _mm256_castpd256_pd128(vec);
    __m128d vhigh = _mm256_extractf128_pd(vec, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to 
}

void print_vec(__m256d var) {
    printf("%f %f %f %f \n", 
           var[0], var[1], var[2], var[3]);
}


/***************************************** IMPLEMENTATIONS *******************************************/


void exact_shapley_base(void *context_ptr) {
    /* exact_shapley_base   combined_knn_shapley_opt*/

    // Call unoptimzed base KNN
    knn_exact_base(context_ptr);

    // Call default implemention of shapley
    compute_single_unweighted_knn_class_shapley(context_ptr);
}


void combined_knn_shapley_opt1(void *context_ptr) {
    /* opt1: just combined */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;

    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;

    double inv_size_x_trn = 1.0/size_x_trn;

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature++) {
                curr_dist += 
                pow(context->x_trn[i_trn*context->feature_len + i_feature] - 
                        context->x_tst[i_tst*context->feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        // Sanity check in order to compare with python
        // debug_print("dist_gt:\n");
        //     for (int j = 0; j<10;j++) {
        //         debug_print("%f, ", dist_gt[j]);
        //     }
        //     debug_print("\n");

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // This can be eliminated later - only here to
        // give us testing/debug information on errors / correctness tests
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));

        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = sorted_indexes[size_x_trn-1];
        double y_tst_j = y_tst[i_tst];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == y_tst_j) ? 1.0 : 0.0;
        sp_gt[i_tst*size_x_trn + offset] = tmp * inv_size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            // debug_print("    i=%d\n", i);
            int x_test_knn_gt_i = sorted_indexes[i];
            int x_test_knn_gt_i_plus_one = sorted_indexes[i+1];

            double s_j_alpha_i_plus_1 = sp_gt[i_tst*size_x_trn + x_test_knn_gt_i_plus_one];
            double difference = (double)(y_trn[x_test_knn_gt_i] == y_tst_j) - 
                                        (double)(y_trn[x_test_knn_gt_i_plus_one] == y_tst_j);
            double min_K_i = K < i+1 ? K : i+1;

            // debug_print("      s_j=%f\n", s_j_alpha_i_plus_1);
            // debug_print("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // debug_print("      min_=%f\n", min_K_i);

            sp_gt[i_tst*size_x_trn + sorted_indexes[i]] = s_j_alpha_i_plus_1 + (difference * inv_K) * (min_K_i / (i+1));
        }
    }
}


void combined_knn_shapley_opt2(void *context_ptr) {
    /* opt2: Blocked Knn + normal integratet shapley + normal integrated sorting*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 12;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 4 == 0);

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=4){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;

                        dist[i1*train_length + j1] += acc_sum1 + acc_sum2;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    double t = sqrt(dist[i2*train_length + j2]);
                    dist[i2*train_length + j2] = t;
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                double min_K_i = K < sj+1 ? K : sj+1;
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * inv_K) * (min_K_i / (sj+1));                
            }
        }
    }
}

void combined_knn_shapley_opt3(void *context_ptr) {
    /* opt3: Blocked Knn + optimized integratet shapley + normal integrated sorting*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 12;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 4 == 0);

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=4){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;

                        dist[i1*train_length + j1] += acc_sum1 + acc_sum2;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    double t = sqrt(dist[i2*train_length + j2]);
                    dist[i2*train_length + j2] = t;
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt4(void *context_ptr) {
    /* opt4: Blocked Knn + optimized integratet shapley + normal integrated sorting*/
    /* 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        double dist_acc5 = 0;
                        double dist_acc6 = 0;
                        double dist_acc7 = 0;
                        double dist_acc8 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            double a4 = x_tst[i1*f_length + k1 + 4];
                            double b4 = x_trn[j1*f_length + k1 + 4];
                            double a5 = x_tst[i1*f_length + k1 + 5];
                            double b5 = x_trn[j1*f_length + k1 + 5];
                            double a6 = x_tst[i1*f_length + k1 + 6];
                            double b6 = x_trn[j1*f_length + k1 + 6];
                            double a7 = x_tst[i1*f_length + k1 + 7];
                            double b7 = x_trn[j1*f_length + k1 + 7];

                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                            dist_acc5 += (a4-b4)*(a4-b4);
                            dist_acc6 += (a5-b5)*(a5-b5);
                            dist_acc7 += (a6-b6)*(a6-b6);
                            dist_acc8 += (a7-b7)*(a7-b7);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;
                        double acc_sum3 = dist_acc5 + dist_acc6;
                        double acc_sum4 = dist_acc7 + dist_acc8;

                        double acc_sum12 = acc_sum1 + acc_sum2;
                        double acc_sum34 = acc_sum3 + acc_sum4;

                        dist[i1*train_length + j1] += acc_sum12 + acc_sum34;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt5(void *context_ptr) {
    /* opt5: Vectorized: Only most inner loop*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;


    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once
                        __m256d dist_acc03 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03 = _mm256_loadu_pd((double *)(x_trn + j1*f_length + k1));
                            __m256d a47 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47 = _mm256_loadu_pd((double *)(x_trn + j1*f_length + k1 + 4));

                            __m256d a_sub_b_03 = _mm256_sub_pd(a03, b03);
                            __m256d a_sub_b_47 = _mm256_sub_pd(a47, b47);

                            __m256d a_sub_b_03_2 = _mm256_mul_pd(a_sub_b_03, a_sub_b_03);
                            __m256d a_sub_b_47_2 = _mm256_mul_pd(a_sub_b_47, a_sub_b_47);

                            dist_acc03 = _mm256_add_pd(dist_acc03, a_sub_b_03_2);
                            dist_acc47 = _mm256_add_pd(dist_acc47, a_sub_b_47_2);
                        }
                        __m256d dist_acc07 = _mm256_add_pd(dist_acc03, dist_acc47);

                        dist[i1*train_length + j1] += (double) vec_sum(dist_acc07);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt6(void *context_ptr) {
    /* opt6: Vectorized: Most inner + loop unrolling of second-most-inner*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_loadu_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_storeu_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}


void combined_knn_shapley_opt7(void *context_ptr) {
    /* opt7: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_loadu_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_storeu_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_loadu_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_storeu_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}


void combined_knn_shapley_opt8(void *context_ptr) {
    /* opt8: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    /* Additionally, rely on aligned memory accesses */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 32;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt9(void *context_ptr) {
    /* opt9: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    /* Additionally, rely on aligned memory accesses */
    /* FIXED BLOCK SIZE OF 8 */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 8;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){

                        /* LOADS */
                        int k1 = k;
                        int j1_vec_e1 = j1;
                        int j1_vec_e2 = j1 + 1;
                        int j1_vec_e3 = j1 + 2;
                        int j1_vec_e4 = j1 + 3;
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        // Entry 1
                        __m256d a03 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                        __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + j1_vec_e1*f_length + k1));
                        __m256d a47 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                        __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + j1_vec_e1*f_length + k1 + 4));
                        // Entry 2
                        __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + j1_vec_e2*f_length + k1));
                        __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + j1_vec_e2*f_length + k1 + 4));
                        // Entry 3
                        __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + j1_vec_e3*f_length + k1));
                        __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + j1_vec_e3*f_length + k1 + 4));
                        // Entry 4
                        __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + j1_vec_e4*f_length + k1));
                        __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + j1_vec_e4*f_length + k1 + 4));
                        // Location to store
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);

                        // COMPUTE
                        // Entry 1
                        __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03, b03_e1);
                        __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47, b47_e1);
                        __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                        __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                        dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                        dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);
                        // Entry 2
                        __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03, b03_e2);
                        __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47, b47_e2);
                        __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                        __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);
                        dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                        dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);
                        // Entry 3
                        __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03, b03_e3);
                        __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47, b47_e3);
                        __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                        __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);
                        dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                        dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);
                        // Entry 4
                        __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03, b03_e4);
                        __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47, b47_e4);
                        __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                        __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);
                        dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                        dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);
                        // Sum
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );
                        // Since its a "+=" we need to load->add->store, instead of only store
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);

                        // STORE
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}

void combined_knn_shapley_opt10(void *context_ptr) {
    /* opt10: Based on Opt8, now force FMA and unroll better */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 32;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e1 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e2 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e2 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e3 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e3 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e4 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e4 = _mm256_set1_pd(0.0);

                        //preload for later
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d a47_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));

                            __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + (j1+0)*f_length + k1));
                            __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + (j1+0)*f_length + k1 + 4));
                            __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + (j1+1)*f_length + k1));
                            __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + (j1+1)*f_length + k1 + 4));
                            __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + (j1+2)*f_length + k1));
                            __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + (j1+2)*f_length + k1 + 4));
                            __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + (j1+3)*f_length + k1));
                            __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + (j1+3)*f_length + k1 + 4));

                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            dist_acc03_e1 = _mm256_fmadd_pd(a_sub_b_03_e1, a_sub_b_03_e1, dist_acc03_e1);
                            dist_acc47_e1 = _mm256_fmadd_pd(a_sub_b_47_e1, a_sub_b_47_e1, dist_acc47_e1);

                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e1, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e1, b47_e2);
                            dist_acc03_e2 = _mm256_fmadd_pd(a_sub_b_03_e2, a_sub_b_03_e2, dist_acc03_e2);
                            dist_acc47_e2 = _mm256_fmadd_pd(a_sub_b_47_e2, a_sub_b_47_e2, dist_acc47_e2);

                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e1, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e1, b47_e3);
                            dist_acc03_e3 = _mm256_fmadd_pd(a_sub_b_03_e3, a_sub_b_03_e3, dist_acc03_e3);
                            dist_acc47_e3 = _mm256_fmadd_pd(a_sub_b_47_e3, a_sub_b_47_e3, dist_acc47_e3);

                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e1, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e1, b47_e4);
                            dist_acc03_e4 = _mm256_fmadd_pd(a_sub_b_03_e4, a_sub_b_03_e4, dist_acc03_e4);
                            dist_acc47_e4 = _mm256_fmadd_pd(a_sub_b_47_e4, a_sub_b_47_e4, dist_acc47_e4);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}

void combined_knn_shapley_opt(void *context_ptr) {
    /* opt11: Based on Opt10
    Redefine input for shapley computation
    Instead of sorting indexes and accessing y_trn through pointer chasing
    Try to use sorted y_trn array directly. 
    Hopefully the shapley computation can then just vectorize all the things

    --> Is working, however indexes of sortex_indexes array are needed to insert shapley value at correct test point.
    If we are interested in shapley values only, we can insert the values for the correct row in whatever order
    then Just summ all up and divide by the number of test points.

    This would then output the shapley vector, instead of the shapley matrix
    But I guess the authors of the paper are interested in the shapley vector...
     */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 32;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    printf("\n-------------------------------\nRunning combined_knn_shapley_opt\n");

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e1 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e2 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e2 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e3 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e3 = _mm256_set1_pd(0.0);
                        __m256d dist_acc03_e4 = _mm256_set1_pd(0.0);
                        __m256d dist_acc47_e4 = _mm256_set1_pd(0.0);

                        //preload for later
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d a47_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));

                            __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + (j1+0)*f_length + k1));
                            __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + (j1+0)*f_length + k1 + 4));
                            __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + (j1+1)*f_length + k1));
                            __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + (j1+1)*f_length + k1 + 4));
                            __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + (j1+2)*f_length + k1));
                            __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + (j1+2)*f_length + k1 + 4));
                            __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + (j1+3)*f_length + k1));
                            __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + (j1+3)*f_length + k1 + 4));

                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            dist_acc03_e1 = _mm256_fmadd_pd(a_sub_b_03_e1, a_sub_b_03_e1, dist_acc03_e1);
                            dist_acc47_e1 = _mm256_fmadd_pd(a_sub_b_47_e1, a_sub_b_47_e1, dist_acc47_e1);

                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e1, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e1, b47_e2);
                            dist_acc03_e2 = _mm256_fmadd_pd(a_sub_b_03_e2, a_sub_b_03_e2, dist_acc03_e2);
                            dist_acc47_e2 = _mm256_fmadd_pd(a_sub_b_47_e2, a_sub_b_47_e2, dist_acc47_e2);

                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e1, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e1, b47_e3);
                            dist_acc03_e3 = _mm256_fmadd_pd(a_sub_b_03_e3, a_sub_b_03_e3, dist_acc03_e3);
                            dist_acc47_e3 = _mm256_fmadd_pd(a_sub_b_47_e3, a_sub_b_47_e3, dist_acc47_e3);

                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e1, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e1, b47_e4);
                            dist_acc03_e4 = _mm256_fmadd_pd(a_sub_b_03_e4, a_sub_b_03_e4, dist_acc03_e4);
                            dist_acc47_e4 = _mm256_fmadd_pd(a_sub_b_47_e4, a_sub_b_47_e4, dist_acc47_e4);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];

            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            // But you need it to pass correctness... comment out and loose, I dare you ;)
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            // int* sorted_indexes = (int*)malloc(train_length * sizeof(int) + train_length * sizeof(double));
            double* combined_sorted_indexes = (double*)malloc(2 * train_length * sizeof(double));
            double* sorted_y_trn = &combined_sorted_indexes[train_length];
            // memcpy((void*)combined_sorted_indexes, (void*)dist_gt_row, train_length * sizeof(double) - 8);
            // memcpy((void*)sorted_y_trn, (void*)y_trn, train_length * sizeof(double) - 8);

            for(int i = 0; i<train_length; i++) {
                // dist_gt_row[i] = train_length-i;
                // y_trn[i] = 10*(train_length-i);
                combined_sorted_indexes[i] = dist_gt_row[i];
                sorted_y_trn[i] = y_trn[i];
            }

            // for(int i = 0; i<train_length; i++) {
            //     combined_sorted_indexes[i] = train_length-i;
            //     sorted_y_trn[i] = 10*(train_length-i);
            //     // combined_sorted_indexes[i] = dist_gt_row[i];
            //     // sorted_y_trn[i] = y_trn[i];
            // }
         
            // printf("\nBefore sort: combined sorted indexes:\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", combined_sorted_indexes[j]);
            // }
            // printf("\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", sorted_y_trn[j]);
            // }
            // printf("\n");

            quicksort(combined_sorted_indexes, (size_t)train_length, sizeof(double));

            // printf("After sort: combined sorted indexes:\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", combined_sorted_indexes[j]);
            // }
            // printf("\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", sorted_y_trn[j]);
            // }
            // printf("\n");

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;

            // printf("correct y_trn access:\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", y_trn[sorted_indexes[j]]);
            // }
            // printf("\n");
            // printf("actual  y_trn access:\n");
            // for (int j = 0; j < train_length; j++){
            //     printf("%f ", sorted_y_trn[j]);
            // }
            // printf("\n\n");
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(sorted_y_trn[sj] == y_test_j) - 
                                            (double)(sorted_y_trn[sj+1] == y_test_j);
                // double actual_difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                //                             (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}
