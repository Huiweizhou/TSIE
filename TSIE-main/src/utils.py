import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser('Interface for Biomedical Hypothesis Generation')

    # select dataset and training mode
    # parser.add_argument('--data', type=str, help='data sources to use',
    #                     choices=['Immunotherapy', 'Virology', 'Neurology'],
    #                     default='Immunotherapy')
    parser.add_argument('--data_path', default='../data/Immunotherapy', type=str, help='The dataset after processed')
    parser.add_argument('--log_path', type=str, default='./logs/Imm_TID', help='Path to save the logs')
    parser.add_argument('--model_save_path', type=str, default='./Models/Imm_TID', help='Path to save the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='whether use gpu')

    parser.add_argument('--model', type=str, default='TID', help='which model to train')
    parser.add_argument('--agg_type', type=str, default='gcn', help='graphsage aggragate methods')
    parser.add_argument('--backbone', type=str, default='Sage', help='Gnn Backbone')
    parser.add_argument('--test_type', type=str, default='full', help='full, transductive, inductive')
    parser.add_argument('--batch_size', type=int, default=2048, help='pairs of one batch')
    parser.add_argument('--nhead', type=int, default=4, help='attn heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--nhead_2', type=int, default=4, help='attn heads')
    parser.add_argument('--num_layers_2', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout_trans', type=float, default=0.6, help='dropout rate')
    parser.add_argument('--input_dim', type=int, default=300, help='input_dim')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--output_dim', type=int, default=1, help='output_dim')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--sample_size_1', type=int, default=20, help='sample neighbors in each layer')
    parser.add_argument('--sample_size_2', type=int, default=10, help='sample neighbors in each layer')
    parser.add_argument('--rnn_wnd', type=int, default=5, help='Two attention modules windows size')
    parser.add_argument('--attn_wnd', type=int, default=5, help='Two attention modules windows size')

    parser.add_argument('--year_start', type=int, default=1950, help='start')
    parser.add_argument('--year_end', type=int, default=2010, help='end')
    parser.add_argument('--year_interval', type=int, default=10, help='interval')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv
