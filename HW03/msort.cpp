
#include <cstddef>
#include <omp.h>
#include <vector>
#include <iostream>


void merge_array ( int* arr, std::size_t left , std::size_t mid , std::size_t right ) {


    std::size_t n1 = mid - left + 1 ;  
    std::size_t n2 = right - mid; 
   
    std::vector<int> Larr(n1);
    std::vector<int> Rarr(n2);
     
    for ( std::size_t i = 0 ; i < n1 ; ++i)  {
      
        Larr[i] = arr[left + i];
        
    }
 
    for ( std::size_t j = 0 ; j < n2 ; ++j)  {
      
        Rarr[j] = arr[mid +1 + j];
    }


    std::size_t i = 0 ; 
    std::size_t j = 0 ; 
    std::size_t k = left ; 
 

    while ( i < n1 && j < n2 ) {
       if (Larr[i] <= Rarr[j]) {
         arr[k] = Larr[i] ;
          ++i ;
       } else {
         arr[k] = Rarr[j] ;
          ++j ;
       }

      ++k;

    }

    while ( i < n1) {
       arr[k] = Larr[i];
       ++i;
       ++k;
    }

    while ( j < n2) {
       arr[k] = Rarr[j];
       ++j;
       ++k;
    }


}

void bubble_sort(int *arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        // for round 1 to n-1
        bool swapped = false;

        for (int j = 0; j < n - i; j++)
        {

            // process element till n-i th index
            if (arr[j] > arr[j + 1])
            {
               std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false)
        {
            // already sorted
            break;
        }
    }
}





void merge_sort ( int* arr, std::size_t left ,  std::size_t right , const std::size_t threshold ) {

 //  std::size_t n = right - left + 1 ;
   std::size_t mid = left + (right - left) / 2 ; 

/*
   for (int i=left; i < right +1; i++){
      std::cout << arr[i] << ",";
      
   } 
*/
   //std::cout << "\n" << arr[left] << "," << arr[right]<< std::endl;
  if ((right - left + 1) >= threshold) {
     
  #pragma omp task shared (arr)  
     merge_sort ( arr , left , mid ,threshold);
  }
  else {
    bubble_sort (arr + left , mid -left + 1 ) ;
  }


  if ( right - mid >= threshold ) {

  #pragma omp task shared (arr)  
     merge_sort ( arr , mid+1, right ,threshold); 

  }
  else {
    bubble_sort (arr + mid + 1  , right - mid ) ;

  }
  #pragma omp taskwait
     merge_array ( arr , left , mid ,right);


 /*  std::cout <<  "After merge" << std::endl;
   for (int i=left; i < right +1; i++){
      std::cout << arr[i] << ",";
      
   } 

   std::cout << std::endl;

*/

/*
   } 
   else {

    // normal sequential sort , bubble sort
      for (std::size_t i = left ; i < n ; ++i) {
          for (std::size_t j = 0 ; j < n - i ; ++j) {              
              if (arr[j] > arr[j+1]) {
                std::swap(arr[j],arr[j+1]);
              }
          }
      }
 
   }
}

*/
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {


  merge_sort(&arr[0],0,n-1,threshold) ;



}

