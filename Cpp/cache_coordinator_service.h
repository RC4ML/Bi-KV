#ifndef CACHE_COORDINATOR_SERVICE_H
#define CACHE_COORDINATOR_SERVICE_H

#include <grpcpp/grpcpp.h>
#include "TaskInfo.grpc.pb.h"  // 由 protoc 生成的头文件

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using TaskInfo::CacheCoordinatorService;
using TaskInfo::TaskInfoList;
using TaskInfo::ComfirmationMessage;
using TaskInfo::StartRequest;
using TaskInfo::Empty;

class CacheCoordinatorServiceImpl final : public CacheCoordinatorService::Service {
public:
    CacheCoordinatorServiceImpl() {}
    
    // 实现 ReceiveTasksFromInferWorker RPC
    Status ReceiveTasksFromInferWorker(ServerContext* context, 
                                     const TaskInfoList* request, 
                                     Empty* response) override;

    // 实现 PollBatchFromInferWorker RPC
    Status PollBatchFromInferWorker(ServerContext* context, 
                                  const TaskInfoList* request, 
                                  ComfirmationMessage* response) override;

    // 实现 StartProcessRequest RPC
    Status StartProcessRequest(ServerContext* context, 
                             const StartRequest* request, 
                             Empty* response) override;

    // 实现 ShutDown RPC
    Status ShutDown(ServerContext* context, 
                   const Empty* request, 
                   Empty* response) override;

private:
    // 这里可以添加服务需要的成员变量
    bool is_running_ = true;  // 控制服务状态
};

#endif  // CACHE_COORDINATOR_SERVICE_H