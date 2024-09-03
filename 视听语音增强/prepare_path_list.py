dataset_path = '/data/ShenXiwen/LAVSE-master/Grid数据集/'

train_spk_list = [88]
val_spk_list   = [88] 
test_spk_list = [14]

snr_test_list = [-5, -2, 1, 4, 7]

def prepare_train_path_list(train_spk_list, dataset):
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in train_spk_list]

    train_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:

            train_path_list.extend([
            (spk_path + 'audio/clean/train/', spk_path + 'audio/noisy/train/', spk_path + 'video-32×32/')#audio_stftpt  64×64 32×32 face   video
            ])

    return train_path_list

def prepare_val_path_list(val_spk_list, dataset):
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in val_spk_list]

    val_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:

            val_path_list.extend([
            (spk_path + 'audio/clean/val/', spk_path + 'audio/noisy/val/', spk_path + 'video-32×32/')
            ])

    return val_path_list

def prepare_test_path_list(test_spk_list, snr_test_list, dataset):#, noise_test_list
    test_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in test_spk_list]

    test_path_list = []
    clean_path_list = []

    if dataset == 'AV_enh':
        for spk_path in test_spk_path_list:
            #for noise in noise_test_list:
            for snr in snr_test_list:
                snrdb =str(snr) + 'dB'   #.replace('-', 'n')     noise + '/' + 
                # ('', noisy_test_path, lip_path)
                test_path_list.extend([
                ('', spk_path + 'audio/不匹配_noisy/test_' + snrdb + '/', spk_path + 'video-32×32/')
                ])

            # ('', clean_test_path, lip_path)
            clean_path_list.extend([
            ('', spk_path + 'audio/clean/', spk_path + 'video-32×32/')
            ])

    return test_path_list, clean_path_list

def prepare_path_list(train_spk_list, val_spk_list, test_spk_list, snr_test_list, dataset):#, noise_test_list
    train_path_list = prepare_train_path_list(train_spk_list, dataset)
    val_path_list = prepare_val_path_list(val_spk_list,dataset)
    test_path_list, clean_path_list = prepare_test_path_list(test_spk_list, snr_test_list, dataset)#, noise_test_list

    return train_path_list, val_path_list, test_path_list, clean_path_list
