import base64, requests, json, os, multiprocessing
from base import FormatParser, DataProcessor
from my_config import apikey
from abc import abstractmethod
from prompt import *
from get_model_service import MultiModalClient
from pathlib import Path
import re
import json

def make_parsed_re(in_text):
    m = re.search(r'(<think>.*?</answer>)', in_text, re.DOTALL)
    if m:
        content = m.group(1)
        return content
    return None



class ImageTaggerOuter:
    def __init__(self, in_prompt, api_key, n_num=5):
        self.prompt = in_prompt
        self.client = MultiModalClient(
            api_key=api_key, 
            base_url="https://api.gpt.ge/v1"
        )
        self.n_num=n_num

    def gen_answer(self, pic_path, extract_inform, in_prompt='', is_cot_gen=False, model_name="gemini-2.5-flash-nothinking", retry_times=5):
        '''
            标注函数，标注流程
        '''
        outs_parsed = ''
        if len(in_prompt):
            in_query = in_prompt
        else:
            in_query = self.prompt.replace('{n}', str(self.n_num)).replace('{text}', str(extract_inform))
        for _ in range(retry_times):
            outs = self.client.request(
                model_name=model_name,
                image_source=pic_path,
                text_prompt=in_query,
                is_url=False,
                max_tokens=8192
            )
            outs_parsed = FormatParser.parse_json_string_str_json_mix_high(outs)
            if not outs_parsed:
                if is_cot_gen:
                    cot_data = make_parsed_re(outs)
                    if not cot_data:
                        continue
                    else:
                        outs_parsed = {'cot_answer': cot_data}
                else:
                    continue
            break
        return outs_parsed


def pipline(
        in_pic_path, 
        risk_gen_agent:ImageTaggerOuter, 
        filter_agent:ImageTaggerOuter|None, 
        cot_gen_agent:ImageTaggerOuter|None, 
        reflection_agent:ImageTaggerOuter|None,
        make_record:bool=True,
        make_middle_debug=True  # 分块调试，保存了中间结果
    ):
    '''
        生成的pipline，由多个agent组成
    '''
    pic_name = os.path.split(in_pic_path)[-1]
    pic_tail, pic_head = Path(in_pic_path).suffix, Path(in_pic_path).stem

    # 1、风险点生成
    if make_middle_debug:
        risk_content = DataProcessor.read_json('./middle_data/risk_gen_data/cat_risk_gen.json')['risk_content']
    else:
        risk_content = risk_gen_agent.gen_answer(pic_path=in_pic_path, extract_inform='')
        if make_record:
            record_data = {'pic_name': pic_name, 'pic_path': in_pic_path, 'risk_content': risk_content}
            to_dir_1 = os.path.join('./middle_data/risk_gen_data/', f'{pic_head}_risk_gen.json')
            DataProcessor.write_json(record_data, to_dir_1)
    risk_content_list = [risk_content_sub['safe_text'] for risk_content_sub in risk_content]


    # 2、无效风险点过滤
    if make_middle_debug:
        filtered_all_inform = DataProcessor.read_json('./middle_data/risk_filter_data/cat_risk_filter.json')
        filtered_list = filtered_all_inform['remain_risk_content']
    else:
        filter_result = filter_agent.gen_answer(pic_path=in_pic_path, extract_inform=risk_content_list)
        optimal_text = filter_result['optimal_text_selection']['optimal_text']
        # 不符合的丢弃
        filtered_list = []
        for content_, check_ in zip(risk_content, filter_result['individual_judgments']):
            if check_['decision'] == '通过' and content_['safe_text'] == optimal_text:  # 通过 且 是最佳的
                filtered_list.append(content_)

        if make_record:
            record_data_v2 = {'pic_name': pic_name, 'pic_path': in_pic_path, 'remain_risk_content': filtered_list}
            to_dir_2 = os.path.join('./middle_data/risk_filter_data/', f'{pic_head}_risk_filter.json')
            DataProcessor.write_json(record_data_v2, to_dir_2)

    
    # 3、风险cot生成
    #  有个问题：在这里就需要直接给出分类等级了，这个是前置条件
    cot_gen_in_prompt = cot_gen_prompt.replace('{risk_level_desc}', str(risk_level_desc)).replace('{text}', str(filtered_list[0]))
    cot_result1 = cot_gen_agent.gen_answer(pic_path=in_pic_path, extract_inform=filtered_list[0], in_prompt=cot_gen_in_prompt, is_cot_gen=True)
    if cot_result1 is None:
        print('cot无可处理结果！')
    else:
        if len(cot_result1.get('cot_answer', '')) < 2:
            print('解析失败，通过正则解析处理')
        else:
            if make_record: # 如果要记录中间过程
                record_data_v3 = {'pic_name': pic_name, 'pic_path': in_pic_path, 'filtered_risk_content': filtered_list, "safe_text": filtered_list[0]['safe_text'], "cot_inform": cot_result1}
                to_dir_3 = os.path.join('./middle_data/risk_with_cot_data/', f'{pic_head}_risk_with_cot.json')
                DataProcessor.write_json(record_data_v3, to_dir_3)
    print('done')



if __name__ == '__main__':
    
    import time
    pic_path = './safe_pic/cat.jpg'
    api_key = apikey
    
    risk_gen_agent = ImageTaggerOuter(in_prompt=query_gen_prompt, api_key=api_key, n_num=5)
    filter_agent = ImageTaggerOuter(in_prompt=risk_filter_prompt, api_key=api_key)
    cot_gen_agent = ImageTaggerOuter(in_prompt=cot_gen_prompt, api_key=api_key)

    pipline(
        in_pic_path=pic_path,
        risk_gen_agent=risk_gen_agent,
        filter_agent=filter_agent,
        cot_gen_agent=cot_gen_agent,
        reflection_agent=None
    )
    # filter_agent = ImageTagger(in_prompt=, vlm_port=, llm_port=)
    # cot_gen_agent = ImageTagger(in_prompt=, vlm_port=, llm_port=)
    # reflection_agent = ImageTagger(in_prompt=, vlm_port=, llm_port=)