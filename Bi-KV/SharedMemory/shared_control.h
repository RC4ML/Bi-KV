#pragma once
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// CPU共享内存中的控制结构
struct SharedControl {
    sem_t sem_start;     // 启动信号量
    sem_t sem_complete;  // 完成信号量
    int data_ready;      // 数据就绪标志 (0/1)
    int error_code;      // 错误代码
};