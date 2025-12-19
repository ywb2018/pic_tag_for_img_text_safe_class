import requests
import time, json
import numpy as np
import logging, os
# import torch
from tqdm import tqdm
import re

import copy

class DataProcessor:
    @classmethod
    def read_jsonl(self, in_file, keep_json_format=False):
        datas = []
        assert ('txt' in in_file or 'jsonl' in in_file or 'xa' in in_file), '输入文件格式有问题'
        if os.path.isfile(in_file):
            with open(in_file, 'r', encoding='utf-8') as f1:
                for line in f1:
                    if keep_json_format:
                        data_ = line.strip()
                    else:
                        data_ = json.loads(line.strip())
                    datas.append(data_)
        return datas
    
    @classmethod
    def write_list_to_file(self, in_data:list, out_file):
        new_data = [i.strip() + '\n' for i in in_data]
        with open(out_file, 'w', encoding='utf-8') as f1:
            f1.writelines(new_data)

    @classmethod
    def read_jsonl_multi_files(self, in_file_list:list, keep_json_format=False):
        all_datas = []
        for f_ in in_file_list:
            d_ = self.read_jsonl(f_, keep_json_format=keep_json_format)
            all_datas.extend(d_)
        return all_datas

    @classmethod
    def read_data_line(self, in_file):
        with open(in_file, 'r', encoding='utf-8') as f1:
            data = f1.readlines()
        return ''.join(data)
    
    @classmethod
    def read_data_lines_list(self, in_file):
        with open(in_file, 'r', encoding='utf-8') as f1:
            data = f1.readlines()
        return [i.strip() for i in data]

    
    def read_prompt(self, in_file):
        with open(in_file, 'r', encoding='utf-8') as f1:
            data = f1.readlines()
        return [i for i in data if len(i.strip()) > 0]

    @classmethod
    def write_json(self, data, out_file):
        with open(out_file, 'w', encoding='utf-8') as f1:
            json.dump(data, f1, ensure_ascii=False, indent=4)
    
    @classmethod
    def write_jsonl_single(self, data:dict, out_file):
        with open(out_file, 'w', encoding='utf-8') as f1:
            f1.write(json.dumps(data, ensure_ascii=False)+'\n')


    @classmethod
    def write_json_line(self, data, out_file):
        with open(out_file, 'w', encoding='utf-8') as f1:
            for d_ in data:
                try:
                    f1.write(json.dumps(d_, ensure_ascii=False)+'\n')
                except Exception as e:
                    print(e)
                    continue
    @classmethod        
    def read_json(self, in_file:str):
        '''
            读json文件
        '''
        datas = None
        assert ('json' in in_file or 'jsonl' in in_file), '输入文件格式有问题'
        if os.path.isfile(in_file):
            with open(in_file, 'r', encoding='utf-8') as f1:
                datas = json.load(f1)
        return datas



class FormatParser:
    '''
        格式处理
    '''
    @classmethod
    def parse_json_string_str_json_mix(self, in_json_string, language='zh'):
        '''
            解析json string，复杂格式合并的
        '''
        assert language in ['zh', 'en']
        try:  # 解析及内容字段检查
            parse_string = in_json_string.split('```json')[-1].split('```')[0].replace('\n', '')
            
            content_res = json.loads(parse_string)
            return content_res
        except Exception as e:
            return None 
    
    @classmethod
    def parse_json_string_str_json_mix_high(self, in_json_string, language='zh'):
        '''
            解析json string，复杂格式合并的
        '''
        assert language in ['zh', 'en']
        try:  # 解析及内容字段检查
            # parse_string = in_json_string.split('```json')[-1].split('```')[0].replace('\n', '')
            parse_string = re.sub(r'\\(x[0-9a-fA-F]{2})', ' ', in_json_string).split('```json')[-1].split('```')[0].replace('\n', '')
            content_res = json.loads(parse_string)
            return content_res
        except Exception as e:
            return None 
        
    @classmethod
    def extract_final_labels(self, response_text:str):
        try:
            think, answer = response_text.split('</think>')
        except:
            think = ''
            answer = response_text
        return answer, think.replace('<think>', '')
    

    def parse_json_string_only_json(self, in_json_string, language='zh'):
        '''
            解析json string
        '''
        assert language in ['zh', 'en']
        try:  # 解析及内容字段检查
            if language == 'zh':
                parse_string = in_json_string.replace('\n', '').replace(' ', '')
            else:
                parse_string = in_json_string.replace('\n', '')
            content_res = json.loads(parse_string)
            return content_res
        except:
            return None 
    

class LLM:
    def inference(self, prompt, url=None):
        raise NotImplementedError
    
    def adjust_params(self, **kargs):
        self.__dict__.update(kargs)
            



class LlmParameter:
    def __init__(
            self,
            temperature=0.7,
            top_k=100,
            top_p=0.7,
            max_tokens=4096,
            do_sample=True,
            repetition_penalty=1.11,
            max_new_tokens=4096
        ):
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.max_tokens=max_tokens
        self.do_sample=do_sample

        self.repetition_penalty=repetition_penalty
        self.max_new_tokens=max_new_tokens


class SelectedCausalLM(LLM, DataProcessor, FormatParser, LlmParameter):
    def __init__(
            self, 
            model_name,
            bias,
            user_parameter_priority=False, # 用户参数优先
            temperature=0.7,
            top_k=30,
            top_p=0.7,
            repetition_penalty=1.12,
            max_new_tokens=4096,
            max_tokens=8192,
            do_sample=True
        ):
        if user_parameter_priority: # 如果用户参数优先
            super().__init__(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens 
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
            
            # 资源文件，配置的是模型的地址 以及需要使用的gpu数目
            self.resource_dict = self.read_json(os.path.join(os.getcwd(), 'base', 'resource.json'))['models']
            assert model_name in self.resource_dict, '模型名称超出范围或不正确!'

            # 取值
            model_inform_dict = self.resource_dict.get(model_name)
            max_memory_mapping = {}
            [max_memory_mapping.update({i+bias: '80GB'}) for i in range(model_inform_dict.get('gpu'))]
            self.generation_config = GenerationConfig.from_pretrained(model_inform_dict.get('path'), trust_remote_code=True)

            super().__init__(
                temperature=temperature if not self.generation_config.temperature else self.generation_config.temperature,
                top_k=top_k if not self.generation_config.top_k else self.generation_config.top_k,
                top_p=top_p if not self.generation_config.top_p else self.generation_config.top_p,
                max_tokens=max_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty if not self.generation_config.repetition_penalty else self.generation_config.repetition_penalty,
                max_new_tokens=max_new_tokens 
            )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
                model_inform_dict.get('path'),
                trust_remote_code=True,
                cache_dir='/home/bml/.cache/huggingface/modules/transformers_modules/',
                torch_dtype="auto",
                device_map='auto',
                max_memory=max_memory_mapping
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_inform_dict.get('path'), trust_remote_code=True)
        print(f'load model: {model_name} done!')

    def clean_gpu(self):
        '''
            清除gpu的显存占用
        '''
        self.model.cpu()
        del self.model
        import torch
        torch.cuda.empty_cache()


    def inference(self, prompt):
        '''
            推理
        '''
        # 格式转换
        messages = []
        messages.append({"role": "user", "content": prompt})
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, max_length=2048, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        import torch
        with torch.no_grad():
            ## 调用
            output_ids = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=self.max_new_tokens,
                max_length=self.max_tokens,
                repetition_penalty=self.repetition_penalty,
                top_p=self.top_p,
                top_k=self.top_k,
                temperature=self.temperature,
                do_sample=self.do_sample,
                use_cache=True
            )
            response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response



class ProducerConsumer:
    def __init__(
            self,
            consumer_func,
            producer_func,
            llm_class,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=32768,
        ):
        self.consumer_func=consumer_func
        self.producer_func=producer_func
        self.llm_class = llm_class
        self.temperature=temperature
        self.top_p=top_p
        self.repetition_penalty=repetition_penalty
        self.max_tokens=max_tokens
    
    def run(
            self,
            check_file_list,
            batch_size,
            model_port_list,
            MAX_QUEUE_SIZE=6000,
            **kwargs

        ):
        import multiprocessing

        NUM_PRODUCERS = len(check_file_list)
        NUM_CONSUMERS = len(model_port_list)
        # 使用JoinableQueue实现任务跟踪
        if MAX_QUEUE_SIZE:
            queue = multiprocessing.JoinableQueue(maxsize=MAX_QUEUE_SIZE)
        else:
            queue = multiprocessing.JoinableQueue()

        # 启动消费者进程
        consumers = []
        for i in range(NUM_CONSUMERS):
            select_port = model_port_list[i]
            llm_ = self.llm_class(
                port=select_port, 
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            p = multiprocessing.Process(target=self.consumer_func, args=(queue, llm_, f'c_{i}'), kwargs=kwargs)
            p.daemon = True
            p.start()
            consumers.append(p)

        # 启动生产者进程  queue, check_file, bsz, producer_id
        producers = []
        for i in range(NUM_PRODUCERS):
            check_file = check_file_list[i]
            p = multiprocessing.Process(target=self.producer_func, 
                                    args=(queue, check_file, batch_size, f'p_{i}'), kwargs=kwargs)
            p.start()
            producers.append(p)

        # 等待所有生产者完成
        for p in producers:
            p.join()

        # 发送终止信号（每个消费者一个None）
        for _ in range(NUM_CONSUMERS):
            queue.put(None)

        # 等待所有任务完成（包括终止信号）
        queue.join()

        print("\nAll tasks completed!")
        


class Qwen2Local(LLM, DataProcessor, FormatParser):
    def __init__(
            self, 
            port=7790,
            temperature=0.6,
            top_k=20,
            top_p=0.6,
            repetion_penalty=1.05,
            max_new_tokens=4096,
            max_tokens=32768,
            do_sample=True,
            url2="http://127.0.0.1:"
        ):
        super().__init__()

        self.url = url2 + str(port) if len(str(port)) else url2
        self.extra_body={
            "top_k": top_k, 
            "max_tokens": max_tokens, 
            "max_new_tokens": max_new_tokens,
            "top_p": top_p, 
            "do_sample": do_sample,
            "temperature": temperature,
            "repetition_penalty": repetion_penalty,
            "stop": ["#&@%*$"],
            "system": ["你是一个有用的ai助手。"]
        }

    def use_model(self, query, normal=False):
        '''
            最多尝试3次推理,如果use_constrain， 则返回score，否则返回text
        '''
        t = 0
        response = None
        in_data = {"query":query}
        if normal:
            extra_body_copy = copy.deepcopy(self.extra_body)
            del extra_body_copy['system']
            del extra_body_copy['stop']
            in_data.update(extra_body_copy)
        else:
            extra_body_copy = copy.deepcopy(self.extra_body)
            extra_body_copy['system'] = ["你是一个有用的ai助手。"] * len(query)
            in_data.update(extra_body_copy)
        response = requests.post(self.url, json=in_data).json()['response']
        return response

    def inference(self, prompt, loop_nums=6):
        for _ in range(loop_nums): # 循环几次解决问题
            try:
                ans = self.use_model([prompt])  # 调用接口
                if isinstance(ans, str):
                    checks = ans
                elif isinstance(ans, list):
                    checks = ans[0]
                if len(checks) > 10:
                    return ans
                continue
            except Exception as e:
                ans = "api调用失败"
                print(ans, e)
                time.sleep(1)
                continue
        return "api调用失败"
    
    def robust_inference_batch(self, prompt:list, check_func, bsz=8, check_loop_nums=5, **kwargs):
        assert isinstance(prompt, list)
        
        
        answer_list = []
        loops = len(prompt) // bsz + 1
        for loop_ in range(loops):
            start, end = loop_ * bsz, (loop_ + 1) * bsz
            small_batch = prompt[start: end]
            if not small_batch:
                continue
            answer_dict = {}
            all_index, all_index_copy = list(range(len(small_batch))), list(range(len(small_batch)))
            # [answer_dict.update({i:''}) for i in all_index]
            wrong_index = list(range(len(small_batch)))
            for _ in range(check_loop_nums):
                try:
                    small_ans = self.use_model(small_batch) 
                    status, wrong_index_this = check_func(small_ans, **kwargs) # 第2轮的时候 错误的index怎么找
                    wrong_index_this = [wrong_index[x] for x in wrong_index_this] # 找到真实的index值
                    right_index= [i for i in all_index if i not in wrong_index_this and i not in answer_dict] 
                    [answer_dict.update({j: small_ans[wrong_index.index(j)]}) for j in right_index ]  # wrong_index.index(j) 是为了定位回本次推理中的位置
                    if not wrong_index_this or status: # 全部都识别出来了
                        break
                    # 还有错误
                    wrong_index = wrong_index_this
                    small_batch = [prompt[i] for i in wrong_index]
                    time.sleep(1)
                except Exception as e:
                    continue
            right_answer = [answer_dict.get(k) for k in all_index_copy]
            answer_list.extend(right_answer)
        return answer_list


    def inference_batch(self, prompt:list, bsz=8):
        assert isinstance(prompt, list)
        for _ in range(6): # 循环几次解决问题
            try:
                answer_list = []
                loops = len(prompt) // bsz + 1
                for loop_ in range(loops):
                    start, end = loop_ * bsz, (loop_ + 1) * bsz
                    small_batch = prompt[start: end]
                    small_ans = self.use_model(small_batch)  # 调用接口
                    answer_list.extend(small_ans)

                return answer_list
                
            except Exception as e:
                ans = "api调用失败"
                print(ans, e)
                time.sleep(1)
                continue
        return "api调用失败"

    
    def inference_normal(self, prompt, loop_nums=6):
        for _ in range(loop_nums): # 循环几次解决问题
            try:
                ans = self.use_model(prompt, normal=True)  # 调用接口
                if isinstance(ans, str):
                    checks = ans
                elif isinstance(ans, list):
                    checks = ans[0]
                if len(checks) > 10:
                    return ans
                continue
            except Exception as e:
                ans = "api调用失败"
                print(ans, e)
                time.sleep(1)
                continue
        return "api调用失败"



class ProducerConsumer:
    def __init__(
            self,
            consumer_func,
            producer_func,
            infer_class,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=32768,
        ):
        self.consumer_func=consumer_func
        self.producer_func=producer_func
        self.temperature=temperature
        self.infer_class = infer_class
        self.top_p=top_p
        self.repetition_penalty=repetition_penalty
        self.max_tokens=max_tokens
    
    def run(
            self,
            producer_num:int,
            port_list:list,
            in_dir:str,
            done_dir:str,
            out_base:str,
            MAX_QUEUE_SIZE=6000,
            **kwargs

        ):
        import multiprocessing

        NUM_PRODUCERS = producer_num
        NUM_CONSUMERS = len(port_list)
        # 使用JoinableQueue实现任务跟踪
        if MAX_QUEUE_SIZE:
            queue = multiprocessing.JoinableQueue(maxsize=MAX_QUEUE_SIZE)
        else:
            queue = multiprocessing.JoinableQueue()

        # 启动消费者进程
        consumers = []
        for i in range(NUM_CONSUMERS):
            vl_model_this = port_list[i]
            c = multiprocessing.Process(target=self.consumer_func, args=(queue, vl_model_this, f'p_{i}', out_base), kwargs=kwargs)
            c.daemon = True
            c.start()
            consumers.append(c)

        # 启动生产者进程  queue, in_dir, done_dir, producer_id, max_retry=1, **kwargs
        producers = []
        for i in range(NUM_PRODUCERS):
            p = multiprocessing.Process(target=self.producer_func, args=(queue, in_dir, done_dir, f'c_{i}'), kwargs=kwargs)
            p.start()
            producers.append(p)

        # 等待所有生产者完成
        for p in producers:
            p.join()

        # 发送终止信号（每个消费者一个None）
        for _ in range(NUM_CONSUMERS):
            queue.put(None)

        # 等待所有任务完成（包括终止信号）
        queue.join()

        print("\nAll tasks completed!")
        


class Qwen2(LLM, DataProcessor, FormatParser, LlmParameter):
    def __init__(
            self, 
            temperature=0.7,
            top_k=50,
            top_p=0.8,
            max_tokens=4096,
            do_sample=True,
            url="http://25.32.5.255:31002/v1/chat/completions",
        ):
        super().__init__(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            do_sample=do_sample
        )
        self.url = url

    def make_input(self, prompt, history): 
        payload = json.dumps({
        "model": "qwen_72B",
        "messages": self.gen_message(prompt, history),
        "max_tokens": 1024,
        "do_sample": True,
        "temperature": self.temperature,
        "top_p": self.top_p,
        "stream":False   
        })
        return payload

    def inference(self, prompt, url=None, loop_nums=6):
        history=[]
        headers = {
            'Content-Type': 'application/json',
            'Cookie': 'Secure; Secure'
            }
        for _ in range(loop_nums): # 循环几次解决问题
            try:
                payload = self.make_input(prompt, history=history)
                response = requests.request("POST", url if url else self.url , headers=headers, data=payload)
                response.encoding="utf-8"

                recvdata = json.loads(response.content.decode())
                ans = recvdata['choices'][0]['message']['content']
                return ans
            except Exception as e:
                ans = "api调用失败"
                print(ans)
                time.sleep(1)
                continue
        
        return "api调用失败"

    def gen_message(self, prompt, history):
        res = []
        system = '你是一个有用的ai助手。'
        res.append({"role":"system","content":system})
        
        for qurey, ans in history:
            res.append({"role":"user","content":qurey})
            res.append({"role":"assistant","content":ans})
        res.append({"role":"user","content":prompt})
        return res


if __name__ == '__main__':
    query = '请尝试使用推理的方式，一步步给出分析并回答下述问题,要求给出一个明确的二值化的结果：我老公去年8月做了结直肠息肉切除手术，病理结果显示为管状腺瘤，我们现在想买蓝医保长期医疗险，我和我53岁的老公能否一起投保？'

    prompt = """
        你是一个擅长推理的问答助手，请回答我给你的问题。
        # 要求
        1、在回答问题时，你需要分成多个推理步骤或分析思路，多个步骤之间使用分隔符：“\n\n\n”，且这多个步骤之间尽量保持逻辑关系。并在最后给出最终的答案
        2、你给出的答案分为两部分：
            a、解决问题的分析思路或推理逻辑,要求该部分尽可能的详细和细致。每个推理步骤之间请用”\n\n\n“进行分隔
            b、给出的答案部分是针对问题的完整回答，需要有明确的答案
        3、给出答案的格式如下: <thought>分析思路或推理逻辑具体内容</Thought> <Output>答案具体内容</Output
        # 信息输入
        输入的问题是：{input}
        请按照要求给出你认为最正确的答案：
    """


    llm = Qwen2Local(port=7782, max_tokens=8192)
    # llm2 = Qwen2Local(port=7791, max_tokens=8192)

    query  = '下列生活用品所含的主要材料，属于有机合成材料的是（） A．铝合金门框 B．瓷碗 C．塑料盆 D．玻璃钢'
    query = '下列变化过程中，没有发生化学变化的是（　　） A．铝的钝化 B．光合作用 C．碘的升华 D．海水提溴'
    query = '向纯水中加入少量NaHS04 固体，当温度不变时，则该溶液中（　　） A．水电离出的c（H+）减小 B．c（OH-）与c（H-）的乘积增大 C．c（H+）减小D．c（OH-）增大'
    in_ques = prompt.replace('{input}', query)

    out = llm.inference(in_ques)
    # out = llm.inference_normal(in_ques)
    # out2 = llm2.inference(in_ques)

    print('got')

    if 0:
        query = '如果我想一个人做8道菜，请问应该怎么合理安排时间？'

        paser = DataProcessor()

        resource_dict = paser.read_json(os.path.join(os.getcwd(), 'base', 'resource.json'))['models']
        for model_key in resource_dict:
            try:
                print('model: ', model_key)
                model_class = SelectedCausalLM(model_name=model_key, bias=0)
                print('generation config: ', model_class.generation_config)
                response = model_class.inference(query)
                print(response.replace('\n', ''), '\n\n\n')
                model_class.clean_gpu()
            except Exception as e:
                print(e, '\n\n')
                continue
