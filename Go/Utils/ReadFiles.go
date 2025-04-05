package utils

import (
	"bufio"
	"encoding/json"
	"io"
	"log"
	"os"
	"strconv"
)

type TaskInfoJson struct {
	RequestID string `json:"request_id"`
	ID        int    `json:"id"`
	TokenNum  int    `json:"token_num"`
	Index     int    `json:"index"`
	TaskType  string `json:"task_type"`
	Type      string `json:"type"`
	TaskNum   int    `json:"task_num"`
}

func ReadJsonTask(filePath string) ([]TaskInfoJson, error) {
	// 打开 JSON 文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 读取文件内容
	content, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}
	// 解析 JSON 数据
	var taskInfoList []TaskInfoJson
	err = json.Unmarshal(content, &taskInfoList)
	if err != nil {
		return nil, err
	}
	return taskInfoList, nil
}

func ReadText(filePath string) (int, error) {
	// 打开文件
	number := 0
	file, err := os.Open(filePath)
	if err != nil {
		return number, err
	}
	defer file.Close()
	// 创建一个扫描器逐行读取文件
	scanner := bufio.NewScanner(file)
	if scanner.Scan() {
		// 读取第一行内容
		line := scanner.Text()
		// 将字符串转换为整数
		number, err = strconv.Atoi(line)
		if err != nil {
			return number, err
		}
		// 输出结果
	} else {
		log.Fatalf("文件为空或读取失败")
	}
	// 检查扫描过程中是否有错误
	if err := scanner.Err(); err != nil {
		return number, err
	}
	return number, nil
}
