package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	General struct {
		ModelCode        string `yaml:"model_code"`
		LlmRetrievedPath string `yaml:"llm_retrieved_path"`
		DatasetCode      string `yaml:"dataset_code"`
		LogLevel         string `yaml:"log_level"`
		LogFile          string `yaml:"log_file"`
	} `yaml:"general"`

	ProcessTypes struct {
		LLMScheduler     int `yaml:"LLMScheduler"`
		CacheCoordinator int `yaml:"CacheCoordinator"`
		Worker           int `yaml:"Worker"`
		KVCache          int `yaml:"KVCache"`
	} `yaml:"process_types"`

	Grpc struct {
		MasterAddr string `yaml:"master_addr"`
		MasterPort int    `yaml:"master_port"`
	} `yaml:"grpc"`

	Distributed struct {
		MasterAddr string `yaml:"master_addr"`
		MasterPort string `yaml:"master_port"`
	} `yaml:"distributed"`

	KvCache struct {
		MaxWorkers int `yaml:"max_workers"`
		CacheSize  int `yaml:"cache_size"`
		PageSize   int `yaml:"page_size"`
	} `yaml:"kv_cache"`

	Worker struct {
		MaxWorkers int `yaml:"max_workers"`
		CacheSize  int `yaml:"cache_size"`
		PageSize   int `yaml:"page_size"`
	} `yaml:"worker"`
}

// 读取配置文件
func ReadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	return &config, err
}
