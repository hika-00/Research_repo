import os

def load_config_generate(dataset_nm):
    config = {
        'model_path': os.path.join(os.getcwd(), 'Data_tobe_loaded', 'trained_model.pt')
    }

    return config