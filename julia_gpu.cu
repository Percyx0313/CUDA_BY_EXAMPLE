#include <iostream>
#include <cuda_runtime.h>
#include "common/cpu_bitmap.h"
#define DIM 1080
#define MAX_ITER 500
struct cuComplex{
    float r;
    float i;
    __device__ cuComplex(float a ,float b):r(a),i(b){}
    __device__ float magnitude2(void) {return r*r+i*i;}
    __device__ cuComplex operator*(const cuComplex& a){
        return cuComplex(r*a.r-i*a.i,i*a.r+r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a){
        return cuComplex(r+a.r,i+a.i);
    }
};

__device__ int IfJulia(int x,int y){
    const float scale=2;
    float jx=scale * (float)(DIM/2-x)/(DIM/2);
    float jy=scale * (float)(DIM/2-y)/(DIM/2);

    cuComplex c(-0.8,0.156);
    cuComplex a(jx,jy);
    for(int i=0;i<MAX_ITER;i++){
        a=a*a+c;
        if(a.magnitude2()>2){
            return 0;
        }
    }
    return 1;
}

__global__ void render(unsigned char* bitmap){
    int x=blockIdx.x;
    int y=blockIdx.y;
    int tid=x+y*gridDim.x;

    int value=IfJulia(x,y);

    bitmap[tid*4+0]=value*255;
    bitmap[tid*4+1]=0;
    bitmap[tid*4+2]=0;
    bitmap[tid*4+3]=255;

}

int main(){

    CPUBitmap bitmap(DIM,DIM);
    unsigned char *ptr = bitmap.get_ptr();
    unsigned char *dev_bitmap;
    // test the bit map
    // for(int i=0;i<DIM/2;i++){
    //     for(int j=0;j<DIM;j++){
    //         ptr[(i*DIM+j)*4]=255;
    //         ptr[(i*DIM+j)*4+1]=0;
    //         ptr[(i*DIM+j)*4+2]=0;
    //         ptr[(i*DIM+j)*4+3]=255;
    //     }
    // }

    // init the resource for cuda
    cudaMalloc((void**)&dev_bitmap,bitmap.image_size());
    dim3 grid(DIM,DIM);

    render<<< grid,1 >>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost);

    

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);

}