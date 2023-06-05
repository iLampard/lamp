from utils.general import file_uri_reader_processor, file_uri_writer_processor


def encode_event_text(source_data, method='bert'):
    if method == 'bert':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    source_data_ = []
    for event in source_data:
        if method == 'bert':
            event_encode = model.encode(event[-1])
        else:
            event_encode = tokenizer.encode(event[-1], padding="max_length", max_length=60, truncation=True)
            print(len(event_encode))
        event_tuple = (event[0], event[1], event[2], event[3], event[4], event_encode)

        source_data_.append(event_tuple)

    print('encoding finished')
    return source_data_


if __name__ == '__main__':
    source_data_fn = '../../data/gdelt/gdelt.pkl'

    source_data = file_uri_reader_processor(source_data_fn)['data']

    source_data_ = encode_event_text(source_data, method='sparse')

    source_data_to_save = {'num_entity': 2279,
                           'num_rel': 20,
                           'data': source_data_}

    file_uri_writer_processor(source_data_to_save, 'gdelt_sparse.pkl')