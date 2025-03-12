package main

import (
	"fmt"
	"net"

	coordinator "github.com/RC4ML/Bi-KV/CacheCoordinator"
	pb "github.com/RC4ML/Bi-KV/protos"

	"google.golang.org/grpc"
)

func main() {
	rank := 1           // 示例 rank
	masterPort := 50500 // 示例端口
	cacheRanks := []int{0, 1, 2, 3}
	inferRanks := []int{0, 1, 2, 3}

	cc := coordinator.NewCacheCoordinator(rank, masterPort, cacheRanks, inferRanks)
	lis, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", masterPort+rank))
	if err != nil {
		fmt.Printf("Failed to listen: %v\n", err)
		return
	}
	s := grpc.NewServer()
	pb.RegisterCacheCoordinatorServiceServer(s, cc)
	fmt.Println("[Main] Starting gRPC server")
	if err := s.Serve(lis); err != nil {
		fmt.Printf("Failed to serve: %v\n", err)
	}
}
