#include "cache_coordinator_service.h"
#include <iostream>

Status CacheCoordinatorServiceImpl::ReceiveTasksFromInferWorker(ServerContext* context,
                                                              const TaskInfoList* request,
                                                              Empty* response) {
    // 处理接收到的任务列表
    std::cout << "Received " << request->tasks_size() << " tasks from InferWorker" << std::endl;
    for (const auto& task : request->tasks()) {
        std::cout << "Task ID: " << task.id() 
                  << ", Request ID: " << task.request_id() 
                  << ", Token Num: " << task.token_num() << std::endl;
    }
    return Status::OK;  // 返回成功状态
}

Status CacheCoordinatorServiceImpl::PollBatchFromInferWorker(ServerContext* context,
                                                           const TaskInfoList* request,
                                                           ComfirmationMessage* response) {
    // 处理任务批次并返回确认消息
    std::cout << "Polling batch with " << request->tasks_size() << " tasks" << std::endl;
    response->set_msg("Batch received and processed successfully");
    return Status::OK;
}

Status CacheCoordinatorServiceImpl::StartProcessRequest(ServerContext* context,
                                                      const StartRequest* request,
                                                      Empty* response) {
    // 处理启动请求
    std::cout << "StartProcessRequest received with message: " << request->msg() << std::endl;
    return Status::OK;
}

Status CacheCoordinatorServiceImpl::ShutDown(ServerContext* context,
                                           const Empty* request,
                                           Empty* response) {
    // 关闭服务
    std::cout << "Shutting down CacheCoordinatorService" << std::endl;
    is_running_ = false;
    return Status::OK;
}