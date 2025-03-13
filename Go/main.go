package main

import (
	"fmt"
	"log"
	"net"
	"os"

	coordinator "github.com/RC4ML/Bi-KV/CacheCoordinator"
	pb "github.com/RC4ML/Bi-KV/protos"

	"google.golang.org/grpc"
)

func main() {
	rank := 1           // 示例 rank
	masterPort := 50500 // 示例端口
	cacheRanks := []int{0, 1, 2, 3}
	inferRanks := []int{0, 1, 2, 3}
	log.SetOutput(os.Stdout)
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
	s := grpc.NewServer()
	cc := coordinator.NewCacheCoordinator(rank, masterPort, cacheRanks, inferRanks, s)
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
