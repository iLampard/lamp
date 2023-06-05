import os
import openai
import elara
from concurrent.futures import ThreadPoolExecutor

from utils.general import file_uri_reader_processor
from utils.gpt_utils import generate_gdelt_prompt_v2, generate_gdelt_comb_prompt
import dictdatabase as DDB


def read_text_file(fn):
    with open(fn, 'r') as file:
        return file.read()


class EventQuery:
    def __init__(self, api_key, prompt_folder: str, num_prompts: int = 12):
        openai.api_key = api_key
        self.setup_msgs = []

        # process prompts
        system_msgs = []
        prompt_assistant_msgs = []
        prompt_user_msgs = []
        help_msgs = []
        if not os.path.exists(prompt_folder):
            raise RuntimeError(f'Prompt folder is not existed: {prompt_folder}')
        for fn in sorted(os.listdir(prompt_folder)):
            file_path = os.path.join(prompt_folder, fn)
            if not os.path.isfile(file_path):
                continue
            msg = read_text_file(file_path)
            if fn.startswith('system_msg'):
                system_msgs.append(msg)
            elif fn.startswith('prompt_assistant'):
                prompt_assistant_msgs.append(msg)
            elif fn.startswith('prompt_user'):
                prompt_user_msgs.append(msg)
            elif fn.startswith('help_msg'):
                help_msgs.append(msg)

        for msg in system_msgs:
            self.setup_msgs.append({
                'role': 'system',
                'content': msg
            })
        for msg in help_msgs:
            self.setup_msgs.append({
                'role': 'user',
                'content': msg
            })
        for user_msg, assistant_msg in zip(prompt_user_msgs[:num_prompts], prompt_assistant_msgs[:num_prompts]):
            self.setup_msgs.append({
                'role': 'user',
                'content': user_msg
            })
            self.setup_msgs.append({
                'role': 'assistant',
                'content': assistant_msg
            })

    def query(self, msg):
        msg_list = self.setup_msgs + [{
            'role': 'user',
            'content': msg
        }]
        completions = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msg_list,
            stream=False
        )
        print('Usage:', completions['usage']['total_tokens'])
        body = completions['choices'][0]['message']['content']
        return body
        # for i, choice in enumerate(completions['choices']):
        #     print('---------------- choice', i)
        #     print(choice['message']['content'])


def gpt_generate_causal_events(
        db_base_name,
        gpt,
        pred_data,
        source_data,
        inference_type: str = 'relation',
        top_k: int = 5,
        num_threads: int = 16
):
    msg_head = '''Now I give you an effect event, and you give me three to five cause events.\n\n'''

    def _process_one_relation_or_object(idx, rel_or_obj, text):
        try:
            existed_dict = DDB.at(db_base_name, inference_type, key=str(idx)).read()
            if existed_dict is not None and str(rel_or_obj) in existed_dict:
                return
            ret_body = gpt.query(msg_head + text)
            with DDB.at(db_base_name, inference_type).session() as (sess, obj):
                if str(idx) in obj:
                    obj[str(idx)][str(rel_or_obj)] = ret_body
                else:
                    obj[str(idx)] = {
                        str(rel_or_obj): ret_body
                    }
                sess.write()
                print(idx, rel_or_obj)
        except Exception as e:
            print('Error of', str(e))

    executor = ThreadPoolExecutor(max_workers=num_threads)
    init_db = DDB.at(db_base_name, inference_type).read()
    if init_db is None:
        DDB.at(db_base_name, inference_type).create()

    for one_prediction in pred_data:
        idx = one_prediction['original_idx']
        if inference_type == 'relation':
            msg_content_dict = generate_gdelt_prompt_v2(one_prediction, source_data, top_k=top_k, pred_relation=True)
        else:
            msg_content_dict = generate_gdelt_prompt_v2(one_prediction, source_data, top_k=top_k, pred_relation=False)
        for rel_or_obj, content in msg_content_dict.items():
            # check if the relation of the idx is existed
            if str(idx) in init_db and str(rel_or_obj) in init_db[str(idx)]:
                continue
            executor.submit(_process_one_relation_or_object, str(idx), rel_or_obj, content)

    executor.shutdown(wait=True)


def gpt_generate_comb_causal_events(
        db_base_name,
        gpt,
        pred_data,
        source_data,
        top_k: int = 100,
        num_threads: int = 6
):
    msg_head = '''Now I give you an effect event, and you give me three to five cause events.\n\n'''

    def _process_one_relation_or_object(idx, rel_obj, text):
        try:
            existed_dict = DDB.at(db_base_name, str(idx), key=str(rel_obj)).read()
            if existed_dict is not None:
                return
            ret_body = gpt.query(msg_head + text)
            with DDB.at(db_base_name, str(idx)).session() as (sess, ddb):
                ddb[str(rel_obj)] = ret_body
                sess.write()
                print(idx, rel_obj)
        except Exception as e:
            print('Error of', str(e))

    executor = ThreadPoolExecutor(max_workers=num_threads)

    for one_prediction in pred_data:
        idx = one_prediction['original_idx']

        idx_db = DDB.at(db_base_name, str(idx)).read()
        if idx_db is None:
            DDB.at(db_base_name, str(idx)).create()
            idx_db = {}

        msg_content_dict = generate_gdelt_comb_prompt(one_prediction, source_data, top_k=top_k, filter_hit=True)

        for rel_obj, content in msg_content_dict.items():
            # check if the relation of the idx is existed
            if str(idx) in idx_db and str(rel_obj) in idx_db:
                continue
            executor.submit(_process_one_relation_or_object, str(idx), rel_obj, content)

    executor.shutdown(wait=True)


if __name__ == '__main__':
    # predictions draw from base model
    base_model_prediction_fn = '../../logs/ke_anhp_gdelt_test.pkl'
    # original sequence data
    raw_seq_data_fn = '../../data/gdelt/gdelt.pkl'

    gpt = EventQuery(
        api_key='*',
        prompt_folder='gdelt_prompts_simplified',
        num_prompts=10
    )

    pred_data = file_uri_reader_processor(base_model_prediction_fn)
    source_data = file_uri_reader_processor(raw_seq_data_fn)['data']

    gpt_generate_causal_events(
        'gdelt_chatgpt',
        gpt,
        pred_data,
        source_data,
        inference_type='relation',
        top_k=5,
        # inference_type='object',
        # top_k=20,
        num_threads=6
    )
