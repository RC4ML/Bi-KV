package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"

	coordinator "github.com/RC4ML/Bi-KV/CacheCoordinator"
	cfg "github.com/RC4ML/Bi-KV/config"
	pb "github.com/RC4ML/Bi-KV/protos"

	"google.golang.org/grpc"
)

func main() {

	// 处理命令行参数
	configPath := flag.String("config", "../config.yml", "path to config file")
	flag.Parse()

	// 读取配置
	config, err := cfg.ReadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to read config: %v", err)
	}
	rank := 1                            // 示例 rank
	masterPort := config.Grpc.MasterPort // 示例端口
	cacheRanks := make([]int, config.ProcessTypes.KVCache)
	for i := range cacheRanks {
		cacheRanks[i] = i
	}

	inferRanks := make([]int, config.ProcessTypes.Worker)
	for i := range inferRanks {
		inferRanks[i] = i
	}
	log.SetOutput(os.Stdout)
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
	s := grpc.NewServer()
	cc := coordinator.NewCacheCoordinator(rank, masterPort, cacheRanks, inferRanks, config.KvCache.CacheSize, config.KvCache.PageSize, s)
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", masterPort+rank))
	if err != nil {
		fmt.Printf("Failed to listen: %v\n", err)
		return
	}
	pb.RegisterCacheCoordinatorServiceServer(s, cc)
	log.Printf("[Main] Starting gRPC server, listen in port %d\n", masterPort+rank)
	if err := s.Serve(lis); err != nil {
		fmt.Printf("Failed to serve: %v\n", err)
	}
}
