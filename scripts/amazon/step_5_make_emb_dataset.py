
from utils.general import file_uri_reader_processor
from utils.misc import make_samples_for_energy_function_amazon

if __name__ == '__main__':
    # source_data_fn = '../../data/amazon/amazon_v0327.pkl'
    source_data_fn = 'amazon_emb.pkl'
    # pred_data_fn = '../../logs/ke_gdelt_test.pkl'
    pred_data_fn = '../../logs/tpp_test_v0426.pkl'

    source_data = file_uri_reader_processor(source_data_fn)['user_seqs']
    pred_data = file_uri_reader_processor(pred_data_fn)

    make_samples_for_energy_function_amazon(
        gpt_db_name='amazon_chatgpt/type',
        source_data=source_data,
        pred_data=pred_data,
        pred_type='type',
        topk=5,
        # pred_type='object',
        # topk=20,
        ebm_db_name='anhp_amazon_ebm_dataset_v3',
        retro_top_n=5,  # make the sequence longer
        distance_type='bert'
    )
