
#include <cstddef>
#include <omp.h>


void merge_sort ( int* arr,  std::size_t left ,  std::size_t right , cost std::size_t threshold ) {

  
   std::size_t mid = left + (right - left) / 2 ;  
   if ( (left < right) && ((right - left + 1) >= threshold)) {
     
  #pragma omp task shared (arr) if (right - left >= threshold) 
     merge_sort ( arr , left , mid ,threshold); 

  #pragma omp task shared (arr) if (right - left >= threshold) 
     merge_sort ( arr , mid+1, right ,threshold); 
  #pragma omp taskwait
     merge_array ( arr , left , mid ,right);

   } 
   else {

    // normal sequential sort , bubble sort
      for (std::size_t i = 0 ; i < n ; ++i) {
          for (std::size_t j = 0 ; j < n ; ++j) {              
              if (arr[j] > arr[j+1]) {
                std::swap(arr[j],arr[j+1]);
              }
          }
      }
      return ;
 
   }
}


void merge_array ( int* arr, std::size_t left , std::size_t mid , std::size_t right ) {


    std::size_t n1 = mid - left + 1 ;  
    std::size_t n2 = right - left ; 
   
    std::vector<int> Larr(n1);
    std::vector<int> Rarr(n2);
    
  
    for ( std::size_t i = 0 ; i < n1 ; ++i)  {
      
        Larr[i] = arr[left + i];
    }
 
    for ( std::size_t j = 0 ; j < n2 ; ++j)  {
      
        Rarr[j] = arr[left + j];
    }


    std:size_t i = 0 ; 
    std:size_t j = 0 ; 
    std:size_t k = left ; 
 

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

   delete[] Larr;
   delete[] Rarr;

}


void msort(int* arr, const std::size_t n, const std::size_t threshold) {


  merge_sort(arr,n,0,n-1,threshold) ;



}

