package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"strings"

	coordinator "github.com/RC4ML/Bi-KV/CacheCoordinator"
	cfg "github.com/RC4ML/Bi-KV/config"
	pb "github.com/RC4ML/Bi-KV/protos"

	"google.golang.org/grpc"
)

func getRankToIPMapping(hostfilePath string) (map[int]string, error) {
	file, err := os.Open(hostfilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open hostfile: %v", err)
	}
	defer file.Close()

	rankToIP := make(map[int]string)
	scanner := bufio.NewScanner(file)
	rank := 0

	for scanner.Scan() {
		line := scanner.Text()
		// Remove any comments and trim whitespace
		if idx := strings.Index(line, "#"); idx != -1 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var ip string
		var slots int
		_, err := fmt.Sscanf(line, "%s slots=%d", &ip, &slots)
		if err != nil {
			return nil, fmt.Errorf("failed to parse line %q: %v", line, err)
		}

		for range slots {
			rankToIP[rank] = ip
			rank++
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read hostfile: %v", err)
	}

	return rankToIP, nil
}

func getRankToIP(rankToIPGRPC []string) (map[int]string, error) {
	rankToIP := make(map[int]string)
	rank := 0
	for _, line := range rankToIPGRPC {
		var ip string
		var slots int
		_, err := fmt.Sscanf(line, "%s slots=%d", &ip, &slots)
		if err != nil {
			return nil, fmt.Errorf("failed to parse line %q: %v", line, err)
		}

		for range slots {
			rankToIP[rank] = ip
			rank++
		}
	}
	return rankToIP, nil
}

func main() {
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
	// hostfilePath := "../Bi-KV/hostfile"
	// rankToIP, err := getRankToIPMapping(hostfilePath)
	rankToIP, err := getRankToIP(config.Grpc.Slots)
	if err != nil {
		log.Fatalf("Error getting rank to IP mapping: %v", err)
	}
	cc := coordinator.NewCacheCoordinator(rank, masterPort, cacheRanks, inferRanks, config.KvCache.CacheSize, config.KvCache.PageSize, s, rankToIP)
	lis, err := net.Listen("tcp", fmt.Sprintf("%s:%d", rankToIP[rank], masterPort+rank))
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
