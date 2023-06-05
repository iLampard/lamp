
from utils.general import file_uri_reader_processor, ProcessPool
from utils.misc import make_comb_samples_for_energy_function

if __name__ == '__main__':

    # source_data_fn = 'gdelt_emb.pkl'
    source_data_fn = '../../data/gdelt/gdelt.pkl'
    # pred_data_fn = '../../logs/ke_gdelt_test.pkl'
    pred_data_fn = '../../logs/ke_anhp_gdelt_test.pkl'


    source_data = file_uri_reader_processor(source_data_fn)['data']
    pred_data = file_uri_reader_processor(pred_data_fn)

    make_samples_for_energy_function(
        gpt_db_name='gdelt_chatgpt/relation',
        source_data=source_data,
        pred_data=pred_data,
        pred_type='relation',
        topk=5,
        # pred_type='object',
        # topk=20,
        ebm_db_name='ke_anhp_gdelt_bert_ebm_dataset',
        retro_top_n=2,
        distance_type='bert'
    )