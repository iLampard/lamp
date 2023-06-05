import os
import openai
import elara
from concurrent.futures import ThreadPoolExecutor

from utils.general import file_uri_reader_processor
from utils.gpt_utils import generate_gdelt_prompt_amazon
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
        inference_type: str = 'type',
        top_k: int = 5,
        num_threads: int = 16
):
    msg_head = '''Now I give you an effect event, and you give me three to four cause events.\n\n'''

    def _process_one_type_or_time(idx, type_or_date, text):
        try:
            existed_dict = DDB.at(db_base_name, inference_type, key=str(idx)).read()
            if existed_dict is not None and str(type_or_date) in existed_dict:
                return
            ret_body = gpt.query(msg_head + text)
            with DDB.at(db_base_name, inference_type).session() as (sess, obj):
                if str(idx) in obj:
                    obj[str(idx)][str(type_or_date)] = ret_body
                else:
                    obj[str(idx)] = {
                        str(type_or_date): ret_body
                    }
                sess.write()
                print(idx, type_or_date)
        except Exception as e:
            print('Error of', str(e))

    executor = ThreadPoolExecutor(max_workers=num_threads)
    init_db = DDB.at(db_base_name, inference_type).read()
    if init_db is None:
        DDB.at(db_base_name, inference_type).create()

    for prediction_seq in pred_data[-10:]:
        for one_prediction in prediction_seq:
            seq_idx = one_prediction['seq_idx']
            original_idx = one_prediction['original_idx']
            idx = (str(seq_idx), str(original_idx))
            if inference_type == 'type':
                msg_content_dict = generate_gdelt_prompt_amazon(one_prediction, source_data, top_k=top_k, pred_type=True)
            else:
                msg_content_dict = generate_gdelt_prompt_amazon(one_prediction, source_data, top_k=top_k, pred_type=False)
            for rel_or_obj, content in msg_content_dict.items():
                # check if the relation of the idx is existed
                if init_db is not None and str(idx) in init_db and str(rel_or_obj) in init_db[str(idx)]:
                    continue
                executor.submit(_process_one_type_or_time, str(idx), rel_or_obj, content)

    executor.shutdown(wait=True)


if __name__ == '__main__':
    # predictions draw from base model
    base_model_prediction_fn = '../../logs/tpp_amazon_test.pkl'
    # original sequence data
    raw_seq_data_fn = '../../data/gdelt/amazon.pkl'

    gpt = EventQuery(
        api_key='*',
        prompt_folder='amazon_prompts',
        num_prompts=8
    )

    pred_data = file_uri_reader_processor(base_model_prediction_fn)
    source_data = file_uri_reader_processor(raw_seq_data_fn)['user_seqs']

    gpt_generate_causal_events(
        'amazon_chatgpt',
        gpt,
        pred_data,
        source_data,
        inference_type='time',
        top_k=5,
        num_threads=6
    )
