import argparse
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from core.eagleeye import EagleEye

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--model_path', type=str, default='./saved_models/cifar10_mobilenetv2_bn.h5')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--min_rate',type=float, default=0.0)
    parser.add_argument('--max_rate',type=float, default=0.5)
    parser.add_argument('--flops_target', type=float, default=0.5)
    parser.add_argument('--num_candidates', type=int, default=15)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--result_dir', type=str, default='./result')


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parser.parse_args()

    eagleeye_obj=EagleEye(dataset_name=args.dataset_name,
                        model_path=args.model_path,
                        bs=args.bs,
                        epochs=args.epochs,
                        lr=args.lr,
                        min_rate=args.min_rate,
                        max_rate=args.max_rate,
                        flops_target=args.flops_target,
                        num_candidates=args.num_candidates,
                        result_dir=args.result_dir,
                        data_augmentation=args.data_augmentation
                        )

    eagleeye_obj.build()

if __name__ == '__main__':
    main()