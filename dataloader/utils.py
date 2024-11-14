import json
import os.path as osp
import torch
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # template_name = "alpaca"
            template_name = "alpaca_short"
        file_name = osp.join("dataloader", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    
def find_goods_index(input_ids):
    '''针对目前prompt格式找的特征，如果prompt变了可能就得重新找\n
    每个商品前后有<0x0A>_(?)这个序列，遍历到)确认前面是<0x0A>_(则开始记录，遍历到<0x0A>停止'''
    input_length = len(input_ids)
    # attn_mask = torch.zeros(input_length, input_length)
    recording = False
    in_candidate = False
    goods_list = []
    # if-else穷举地狱 或许有空得优化一下
    for i in range(input_length):
        # 11565 -> "▁pool"
        if input_ids[i] == 11565:
            # 往上查五个
            if i-4>=0 and input_ids[i-4]==13 and (input_ids[i-3] == 29907 or input_ids[i-3] == 315) and input_ids[i-2] == 5380 and input_ids[i-1] == 403:
                # 匹配到<0x0A>Candidate_pool/<0x0A>_Candidate_pool
                # print(f"Found <0x0A>Candidate_pool in {i} Start Record")
                in_candidate = True
        # 29897-> ")"
        if input_ids[i] == 29897 and in_candidate:
            # 313-> "_(" 13->"<0x0A>"
            if (i-3>=0 and input_ids[i-2]==313 and (input_ids[i-3]==13 or input_ids[i-3]==29901)) or (i-4>=0 and input_ids[i-3]==313 and (input_ids[i-4]==13 or input_ids[i-4]==29901)):
                # 满足匹配模式<0x0A>_(?)/:_(?) 开始记录
                # 有一个问题，如果输入里面有:怎么办，得替换一下
                recording = True
                # print(f"{i}: Found _(")
                start_ind = i+1
        elif input_ids[i]==13 and recording and in_candidate:
            recording = False
            goods_list.append(list(range(start_ind,i)))
        # 匹配到<0x0A>_User_history/<0x0A>User_history
        if input_ids[i] == 4955:
            if i-2>=0 and input_ids[i-2] == 13 and (input_ids[i-1] == 4911 or input_ids[i-1] == 2659):
                # print(f"Found <0x0A>_User_history in {i} Stop Record")
                in_candidate = False
        # 或者匹配到###▁Response
        if input_ids[i] == 13291:
            if i-2>=0 and input_ids[i-2] == 2277 and input_ids[i-1] == 29937:
                # print(f"Found ###▁Response in {i} Stop Record")
                in_candidate = False
    return goods_list


def find_user_history(input_ids):
    '''找用户历史的长度'''
    input_length = len(input_ids)
    recording = False
    history_length = -1
    for i in range(input_length):
        if input_ids[i] == 4955:
            if i-2>=0 and input_ids[i-2] == 13 and (input_ids[i-1] == 4911 or input_ids[i-1] == 2659):
                # print(f"Found <0x0A>_User_history in {i} Stop Record")
                recording = True
                start_ind = i
        # 或者匹配到###▁Response
        if input_ids[i] == 13291:
            if i-2>=0 and input_ids[i-2] == 2277 and input_ids[i-1] == 29937 and recording:
                recording = False
                history_length = i - start_ind
        if input_ids[i] == 11565:
            # 往上查五个
            if i-4>=0 and input_ids[i-4]==13 and (input_ids[i-3] == 29907 or input_ids[i-3] == 315) and input_ids[i-2] == 5380 and input_ids[i-1] == 403:
                # 匹配到<0x0A>Candidate_pool/<0x0A>_Candidate_pool
                # print(f"Found <0x0A>Candidate_pool in {i} Start Record")
                recording = False
                history_length = i - start_ind
    return history_length
        

def find_goods_index_llama3(input_ids):
    '''
    针对目前Llama3 Tokenizer找的特征，如果prompt变了可能就得重新找\n
    每个商品前后有ĠĊ Ġ( )这个序列，遍历到)确认前面是ĠĊ Ġ(则开始记录，遍历到ĠĊ/ĊĊ停止
    '''
    input_length = len(input_ids)
    # attn_mask = torch.zeros(input_length, input_length)
    recording = False
    in_candidate = False
    goods_list = []
    # if-else穷举地狱 或许有空得优化一下
    for i in range(input_length):
        # 7463: "Ġpool"
        if input_ids[i] == 7463:
            # 往上查两个 720: "ĠĊ"
            if i-2>=0 and ((input_ids[i-1]==65001 and input_ids[i-2]==512) or (input_ids[i-1]==50683 and input_ids[i-2]==720)):
                # 匹配到<0x0A>Candidate_pool/<0x0A>_Candidate_pool
                # print(f"Ġpool in {i} Start Record")
                in_candidate = True
        # 8-> ")"
        if input_ids[i] == 8 and in_candidate:
            # 320: "Ġ(" 720: "ĠĊ" 25:":"
            if i-3>=0 and input_ids[i-2]==320 and (input_ids[i-3]==720 or input_ids[i-3]==25):
                # 满足匹配模式<0x0A>_(?)/:_(?) 开始记录
                # 有一个问题，如果输入里面有:怎么办，得替换一下
                recording = True
                # print(f"{i}: Found _(")
                start_ind = i+1
        # 271: "ĊĊ"
        elif (input_ids[i]==720 or input_ids[i]==271) and recording and in_candidate:
            recording = False
            goods_list.append(list(range(start_ind,i)))
        # 匹配到ĠĊ User Ġhistory/Ċ User Ġhistory
        if input_ids[i] == 3925 and ((input_ids[i-1] == 1502 or input_ids[i-2] == 512) or (input_ids[i-1] == 2724 or input_ids[i-2] == 720)):
            # print(f"Found <0x0A>_User_history in {i} Stop Record")
            in_candidate = False
        # 6075: ĠResponse
        if input_ids[i] == 6075:
            # 14711: "###"
            if i-1>=0 and input_ids[i-1] == 14711:
                # print(f"Found ###▁Response in {i} Stop Record")
                in_candidate = False
    return goods_list


def generate_goods_mask(input_ids,pattern_func = find_goods_index):
    goods_list = pattern_func(input_ids)
    input_length = len(input_ids)
    attn_mask = torch.zeros(input_length, input_length)
    
    for ind,i in enumerate(goods_list):
        # 第一层遍历
        # 得到其他元素的列表
        combined = []
        # 遍历嵌套列表中的每个子列表
        for jnd,j in enumerate(goods_list):
            if jnd != ind:  # 如果不是当前子列表
                combined.extend(j)
        attn_mask[torch.tensor(i).unsqueeze(1), torch.tensor(combined).unsqueeze(0)] = 1
    return attn_mask

def generate_position_ids(input_ids,pattern_func = find_goods_index):
    result = list(range(len(input_ids)))
    goods_list = pattern_func(input_ids)
    # 先改成和第一个商品pos_id一样
    if goods_list[0]:
        pos_id = goods_list[0][0]
    else:
        pos_id = 0
        print("Warning: pos_id set to 0!")
    for sublist in goods_list:
        if sublist:  # 确保子列表非空
            start_ind,end_ind = sublist[0],sublist[-1]
            # 第一种，统一为第一个商品第一个token的pos_id
            # new_values = [pos_id] * (end_ind - start_ind + 1)
            step_num = end_ind - start_ind + 1
            new_values = list(range(pos_id,pos_id+step_num))
            result[start_ind:end_ind+1] = new_values
    return result

def get_excel_column_name(n):
    # Excel列名从1开始计数，所以如果n是从0开始的索引，需要加1
    n += 1
    name = ''
    while n > 0:
        n, r = divmod(n - 1, 26)
        name = chr(r + ord('A')) + name
    return name