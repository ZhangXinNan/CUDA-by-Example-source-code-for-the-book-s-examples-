
第三章    CUDA C简介 
___________
代码地址：
[ZhangXinNan/CUDA-by-Example-source-code-for-the-book-s-examples-](https://github.com/ZhangXinNan/CUDA-by-Example-source-code-for-the-book-s-examples-)




## 3.2 第一个程序

### 3.2.1 hello world
代码：hello_world.cu
* 主机（host)：CPU及系统的内存称为主机。
* 设备（device）：GPU及其内存称为设备。
* 核函数（kernel）：GPU上运行的函数称为核函数。

### 3.2.2 核函数调用
代码：simple_kernel.cu
```c++
__global__ ：告诉编译器，函数应该编译为在设备而不是在主机上运行。
```

### 3.2.3 传递参数
代码：simple_kernel_params.cu
* cudaMolloc()      CUDA运行时在设备上分配内存
* cudaFree()        释放cudaMolloc()分配的内存
* cudaMemcpy()      访问设备上的内存

【注】
（1）不能为主机代码中使用cudaMalloc()分配的指针进行内存读写操作。
（2）主机指针只能访问主机代码中的内存，设备指针只能访问设备代码中的内存。

## 3.3 查询设备
代码：enum_gpu.cu
* cudaGetDeviceCount()      获得CUDA设备的数量。
* cudaDeviceProp            结构体，设备的相关属性。
* cudaGetDeviceProperties() 查询设备的相关信息



输出：
```
   --- General Information for device 0 ---
Name:  GeForce GTX 1070
Compute capability:  6.1
Clock rate:  1695000
Device copy overlap:  Enabled
Kernel execution timeout :  Enabled
   --- Memory Information for device 0 ---
Total global mem:  0
Total constant Mem:  65536
Max mem pitch:  2147483647
Texture Alignment:  512
   --- MP Information for device 0 ---
Multiprocessor count:  16
Shared mem per mp:  49152
Registers per mp:  65536
Threads in warp:  32
Max threads per block:  1024
Max thread dimensions:  (1024, 1024, 64)
Max grid dimensions:  (2147483647, 65535, 65535)
```

## 3.4 设备属性的使用
代码：set_gpu.cu
输出：
```
ID of current CUDA device:  0
ID of CUDA device closest to revision 1.3:  0
```


* cudaChooseDevice()        选择满足条件的某个设备
* cudaSetDevice()           所有的设备操作执行在这个设备上。



