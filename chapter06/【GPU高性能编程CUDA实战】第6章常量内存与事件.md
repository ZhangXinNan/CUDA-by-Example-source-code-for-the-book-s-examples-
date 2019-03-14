
## 6.2 常量内存

```c++
__constant__        //修饰符，常量内存变量
cudaMemcpyToSymbol  // 复制到常量内存
```

当线程束中的所有线程都访问相同的只读数据时，将获得额外的性能提升。在这种数据访问模式中使用常量内存可以节约内存带宽，因为（1）读取操作在半线程束中广播，（2）在芯片上包含了常量内存缓存。


## 6.3 使用事件来测量性能

```c++
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
```



