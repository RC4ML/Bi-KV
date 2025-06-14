package coordinator

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	cfg "github.com/RC4ML/Bi-KV/config"
	pb "github.com/RC4ML/Bi-KV/protos"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// CacheCoordinator 定义调度器的结构体
type CacheCoordinator struct {
	pb.UnimplementedCacheCoordinatorServiceServer
	rank                 int
	masterPort           int
	cacheRanks           []int
	inferRanks           []int
	rankToIP             map[int]string
	kvcacheNum           int32
	requestQueue         chan *pb.TaskInfo // 使用 channel 替代 Python 的 Queue
	finishedCounterTable map[int32]int32
	processFlag          bool
	cpuStateTable        map[int]map[string]string
	lock                 sync.Mutex
	stopLimit            int
	cacheSize            int
	pageSize             int
	pageManager          *MultiPageManager // MultiPageManager 的 Go 实现
	cacheMissDict        map[int32]map[int32]int32
	server               *grpc.Server
}

// NewCacheCoordinator 初始化调度器
func NewCacheCoordinator(rank, masterPort int, cacheRanks, inferRanks []int, server *grpc.Server, rankToIP map[int]string, config *cfg.Config) *CacheCoordinator {
	cc := &CacheCoordinator{
		rank:                 rank,
		masterPort:           masterPort,
		cacheRanks:           cacheRanks,
		inferRanks:           inferRanks,
		rankToIP:             rankToIP,
		kvcacheNum:           int32(len(cacheRanks)),
		requestQueue:         make(chan *pb.TaskInfo, 2000), // 无缓冲的channel作为queue
		finishedCounterTable: make(map[int32]int32),
		cpuStateTable:        make(map[int]map[string]string),
		stopLimit:            50000,
		cacheSize:            config.KvCache.CacheSize,
		pageSize:             config.KvCache.PageSize,
		pageManager:          NewMultiPageManager(config.KvCache.CacheSize, config.KvCache.PageSize, len(cacheRanks), config.Coordinator.GCIntreval, config.Coordinator.TTLINterval), // 初始化 PageManager
		cacheMissDict:        make(map[int32]map[int32]int32),
		server:               server,
	}
	kvcacheNum := int(cc.kvcacheNum)
	for i := range kvcacheNum {
		cc.cpuStateTable[i] = map[string]string{"status": "idle"}
	}
	if config.Coordinator.ReadPreparedData {
		cc.pageManager.ReadPreparedData(config.Coordinator.PrepareDataPath, config.Coordinator.PrepareIndexPath)
	}
	log.Println("[CacheCoordinator] 初始化完成")
	return cc
}

// Strategy 分配策略
func (cc *CacheCoordinator) Strategy(reqID int32) int32 {
	return reqID % cc.kvcacheNum
}

// ReceiveTasksFromInferWorker gRPC 服务方法
func (cc *CacheCoordinator) ReceiveTasksFromInferWorker(ctx context.Context, req *pb.TaskInfoList) (*pb.Empty, error) {
	// log.Printf("[CacheCoordinator] 收到请求，长度为%d\n", len(req.Tasks))
	// cc.pageManager.ShowFreePages()
	for _, task := range req.Tasks {
		task.CacheWorker = int32(cc.Strategy(task.RequestId + task.Id))
		cc.requestQueue <- task
	}
	return &pb.Empty{}, nil
}

// PollBatchFromInferWorker gRPC 服务方法
func (cc *CacheCoordinator) PollBatchFromInferWorker(ctx context.Context, req *pb.TaskInfoList) (*pb.ComfirmationMessage, error) {
	// start := time.Now()
	// log.Println("[CacheCoordinator] Start poll")
	requestToTaskNum := make(map[int32]int32)
	for _, task := range req.Tasks {
		if _, ok := requestToTaskNum[task.RequestId]; !ok {
			requestToTaskNum[task.RequestId] = task.TaskNum
		} else if requestToTaskNum[task.RequestId] != task.TaskNum {
			return nil, fmt.Errorf("conflicting task_num for request id %d", task.RequestId)
		}
	}

	cacheMissDict := make(map[int32]map[int32]int32)
	unfinished := make(map[int32]bool)
	for reqID := range requestToTaskNum {
		unfinished[reqID] = true
	}
	for len(unfinished) > 0 {
		for reqID := range unfinished {
			taskNum := requestToTaskNum[reqID]
			cc.lock.Lock()
			resCounter := cc.finishedCounterTable[reqID]
			if resCounter == taskNum {
				cacheMissDict[reqID] = cc.cacheMissDict[reqID]
				delete(unfinished, reqID)
			}
			cc.lock.Unlock()
		}
	}

	// fmt.Println(cacheMissDict)
	data, err := json.Marshal(cacheMissDict)
	if err != nil {
		return nil, err
	}
	// duration := time.Since(start)
	// log.Printf("[CacheCoordinator] Finish poll time cost:%v\n", duration)
	return &pb.ComfirmationMessage{Msg: string(data)}, nil
}

// StartProcessRequest gRPC 服务方法
func (cc *CacheCoordinator) StartProcessRequest(ctx context.Context, req *pb.StartRequest) (*pb.Empty, error) {
	cc.processFlag = true
	go cc.processRequests() // 在 goroutine 中处理请求
	return &pb.Empty{}, nil
}

// processRequests 处理请求的主逻辑
func (cc *CacheCoordinator) processRequests() {
	fmt.Println("[CacheCoordinator] 开始处理请求")
	idleTimeCounter := 0
	hasExecuted := false

	cc.pageManager.ShowUserCacheCount()


	for cc.processFlag {
		executableRequests := make(map[int][]*pb.TaskInfo)
		for {
			select {
			case taskInfo := <-cc.requestQueue:
				idleTimeCounter = 0
				reqID := taskInfo.RequestId
				cc.lock.Lock()
				if taskInfo.TaskType == SIGNAL_CHECK {
					if _, ok := cc.cacheMissDict[reqID]; !ok {
						cc.cacheMissDict[reqID] = make(map[int32]int32)
					}
					// 查cache
					cacheWorker, pages := cc.pageManager.AccessItem(taskInfo.Id)
					if cacheWorker == -1 {
						cc.cacheMissDict[reqID][taskInfo.Id] = CACHE_MISS
					} else {
						cc.cacheMissDict[reqID][taskInfo.Id] = CACHE_HIT
						// cc.pageManager.pageManagers[cacheWorker].SetProtected(taskInfo.Id)
						taskInfo.TaskType = SIGNAL_SEND
						taskInfo.CacheWorker = cacheWorker
						taskInfo.CachePagesList = pages
					}

					// 测试用 全hit
					// cacheWorker, pages := cc.pageManager.LoadItem(taskInfo.Id, int(taskInfo.TokenNum), taskInfo.Weight, taskInfo.Type)
					// taskInfo.CacheWorker = cacheWorker
					// taskInfo.CachePagesList = pages
					// cc.cacheMissDict[reqID][taskInfo.Id] = CACHE_HIT
					// taskInfo.TaskType = SIGNAL_SEND

				} else if taskInfo.TaskType == SIGNAL_RECV {
					cacheWorker, pages := cc.pageManager.LoadItem(taskInfo.Id, int(taskInfo.TokenNum), taskInfo.Weight, taskInfo.Type)
					// cc.pageManager.pageManagers[cacheWorker].SetProtected(taskInfo.Id)
					taskInfo.CacheWorker = cacheWorker
					taskInfo.CachePagesList = pages
					cc.pageManager.ShowUserCacheCount()

				}

				cacheWorker := int(taskInfo.CacheWorker)
				if _, ok := cc.finishedCounterTable[reqID]; !ok {
					cc.finishedCounterTable[reqID] = 0
				}
				executableRequests[cacheWorker] = append(executableRequests[cacheWorker], taskInfo)
				hasExecuted = true
				cc.lock.Unlock()
			default:
				if cc.pageManager.loadDuration > 0 {
					// fmt.Printf("load duration:%v evt:%v all:%v\n", cc.pageManager.loadDuration, cc.pageManager.evtDuration, cc.pageManager.allDuration)
					cc.pageManager.loadDuration = 0
					cc.pageManager.allDuration = 0
					cc.pageManager.evtDuration = 0
				}
				goto process
			}
		}
	process:
		if len(executableRequests) == 0 && !hasExecuted {
			continue
		}
		if len(executableRequests) > 0 {
			log.Printf("[CacheCoordinator] start execute")
		}
		startTime := time.Now()
		var wg sync.WaitGroup
		for cacheWorker, tasks := range executableRequests {
			wg.Add(1)
			go func(worker int, taskList []*pb.TaskInfo) {
				defer wg.Done()
				cc.executeRequestBatch(worker, taskList)
			}(cacheWorker, tasks)
		}
		wg.Wait()
		duration := time.Since(startTime)
		if len(executableRequests) > 0 {
			log.Printf("[CacheCoordinator] executeRequestBatch time cost: %v\n", duration)
		}

		cc.lock.Lock()
		if idleTimeCounter > cc.stopLimit && len(cc.requestQueue) == 0 {
			log.Println("[CacheCoordinator] Empty request table. B R E A K")
			cc.sendTerminateSignal()
			cc.lock.Unlock()
			break
		}
		cc.lock.Unlock()

		if len(cc.requestQueue) == 0 {
			idleTimeCounter++
			time.Sleep(500 * time.Microsecond)
		}
	}
	log.Println("[CacheCoordinator] 所有请求处理完成")
}

// executeRequestBatch 执行批量请求
func (cc *CacheCoordinator) executeRequestBatch(cacheWorker int, reqList []*pb.TaskInfo) {
	cacheRank := 2*cacheWorker + KVCACHEOffset // 假设 KVCACHEOffset 已定义
	addr := fmt.Sprintf("%s:%d", cc.rankToIP[cacheRank], cc.masterPort+cacheRank)
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("[CacheCoordinator] Failed to connect: %v\n", err)
		return
	}
	defer conn.Close()

	client := pb.NewKVCacheServiceClient(conn)
	// fmt.Printf("[CacheCoordinator] Try to execute...len=%d\n", len(reqList))
	resp, err := client.ReceiveTasksFromCoordinator(context.Background(), &pb.TaskInfoList{Tasks: reqList})
	if err != nil {
		log.Printf("[CacheCoordinator] Failed to send tasks: %v\n", err)
		return
	}
	// 解除保护
	// for _, i := range reqList {
	// 	if i.TaskType != SIGNAL_CHECK && i.Id != -1 {
	// 		cc.pageManager.pageManagers[cacheWorker].RemoveProtected(i.Id)
	// 	}
	// }
	var confirmationMsg map[int32]int32
	if err := json.Unmarshal([]byte(resp.Msg), &confirmationMsg); err != nil {
		log.Printf("[CacheCoordinator] Failed to unmarshal: %v\n", err)
		return
	}

	cc.lock.Lock()
	for reqID, count := range confirmationMsg {
		cc.finishedCounterTable[reqID] += count
	}
	cc.lock.Unlock()
}

// sendTerminateSignal 发送终止信号
func (cc *CacheCoordinator) sendTerminateSignal() {
	log.Println("[CacheCoordinator] 发送终止信号给所有 KVCache")
	var wg sync.WaitGroup
	for _, rank := range cc.cacheRanks {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			cacheRank := 2*r + KVCACHEOffset // 假设 KVCACHEOffset 已定义
			addr := fmt.Sprintf("%s:%d", cc.rankToIP[cacheRank], cc.masterPort+cacheRank)
			conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				log.Printf("[CacheCoordinator] Failed to connect KVCache %d: %v\n", r, err)
				return
			}
			defer conn.Close()
			client := pb.NewKVCacheServiceClient(conn)
			_, err = client.ShutDown(context.Background(), &pb.Empty{})
			if err != nil {
				log.Printf("[CacheCoordinator] Failed to terminate KVCache %d: %v\n", r, err)
			}
		}(rank)
	}
	log.Println("[CacheCoordinator] 发送终止信号给所有 Infer Worker")
	for _, rank := range cc.inferRanks {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			workerRank := 2*r + INFEROffset
			addr := fmt.Sprintf("%s:%d", cc.rankToIP[workerRank], cc.masterPort+workerRank)
			conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				log.Printf("[CacheCoordinator] Failed to connect Infer Worker %d: %v\n", r, err)
				return
			}
			defer conn.Close()
			client := pb.NewInferWorkerServiceClient(conn)
			_, err = client.ShutDown(context.Background(), &pb.Empty{})
			if err != nil {
				log.Printf("[CacheCoordinator] Failed to terminate Infer Worker %d: %v\n", r, err)
			}
		}(rank)
	}
	wg.Wait()
	log.Println("[CacheCoordinator] 终止信号已发送...停止Coordinator")
	cc.server.Stop()
}

func (cc *CacheCoordinator) ReceiveTasksFromScheduler(ctx context.Context, req *pb.TaskInfoList) (*pb.ComfirmationMessage, error) {
	// cacheMiss 统计一批task的情况 user为user miss的token数 item为item miss的token数
	// log.Printf("[CacheCoordinator] 收到调度查询请求，长度为%d\n", len(req.Tasks))
	cacheMiss := make(map[int32]map[string]int32)
	itemMiss := make(map[int32]int32)
	userMiss := make(map[int32]int32)
	cc.lock.Lock()
	for _, task := range req.Tasks {
		_, hit := cc.pageManager.AccessItem(task.Id)
		if _, ok := cacheMiss[task.RequestId]; !ok {
			// log.Printf("%d init", task.RequestId)
			cacheMiss[task.RequestId] = make(map[string]int32)
			userMiss[task.RequestId] = 0
			itemMiss[task.RequestId] = 0
		}
		if hit == nil {
			if task.Type == "user cache" {
				userMiss[task.RequestId] = task.TokenNum
			} else {
				itemMiss[task.RequestId] += task.TokenNum
			}
		}
	}
	cc.lock.Unlock()
	for key := range cacheMiss {
		cacheMiss[key]["user miss"] = userMiss[key]
		cacheMiss[key]["item miss"] = itemMiss[key]
	}
	data, err := json.Marshal(cacheMiss)
	if err != nil {
		return nil, err
	}
	return &pb.ComfirmationMessage{Msg: string(data)}, nil
}

func setToList(set map[int32]struct{}) []int32 {
	list := make([]int32, 0, len(set)) // 预分配切片容量以提高性能
	for key := range set {
		list = append(list, key)
	}
	return list
}

// 定义常量（需根据您的 Python 代码调整）
const (
	SIGNAL_CHECK  = int32(4)
	SIGNAL_SEND   = int32(1)
	SIGNAL_RECV   = int32(2)
	CACHE_MISS    = int32(0)
	CACHE_HIT     = int32(1)
	INFEROffset   = 2
	KVCACHEOffset = 3
)
