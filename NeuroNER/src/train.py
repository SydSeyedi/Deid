import os
import tensorflow as tf
import numpy as np
import sklearn.metrics
from evaluate import remap_labels
import pickle
import utils_tf
import codecs
import utils_nlp
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_step(sess, dataset, sequence_number, model, parameters):
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.UNK_TOKEN_INDEX
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: 1-parameters['dropout_rate']
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number, parameters, dataset_filepaths):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type,epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')
    
#
    A=[]
  #  B=[]
    C=[]
  #  ti=[]
  #  chi=[]
  #  tl=[]
  #  lvi=[]
#
    for i in range(len(dataset.token_indices[dataset_type])):
        feed_dict = {
          model.input_token_indices: dataset.token_indices[dataset_type][i],
          model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
          model.input_token_lengths: dataset.token_lengths[dataset_type][i],
          model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
          model.dropout_keep_prob: 1.
        }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
#
        #np.save("SalmanTest/DICT%s%s"%(i,dataset_type),feed_dict)
        #save_path = saver.save(sess,"./SalmanTest/CHECKDIC.ckpt")
        #np.save("SalmanTest/SICT%s%s"%(i,dataset_type),predictions)
        #print("SALMAAAAANNNNNNNNNNNNNN = %s"%feed_dict)
        #print("SALMAAAAAN11111111NNNNNNNNNNNNN = %s"%dataset.token_indices[dataset_type][i])
        #print("SALMAAAAAN22222222NNNNNNNNNNNNN = %s"%dataset.character_indices_padded[dataset_type][i])
        #print("SALMAAAAAN33333333NNNNNNNNNNNNN = %s"%dataset.token_lengths[dataset_type][i])
        #print("SALMAAAAAN44444444NNNNNNNNNNNNN = %s"%dataset.label_vector_indices[dataset_type][i])
  #      ti.append(dataset.token_indices[dataset_type][i])
  #      chi.append(dataset.character_indices_padded[dataset_type][i])
  #      tl.append(dataset.token_lengths[dataset_type][i])
  #      lvi.append(dataset.label_vector_indices[dataset_type][i])
        A.append(unary_scores[1:,:-2][:-1])
  #      B.append(predictions)
#
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

#
        C.append(predictions)
#
        assert(len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]
        if parameters['tagging_format'] == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)
        for prediction, token, gold_label in zip(prediction_labels, dataset.tokens[dataset_type][i], gold_labels):
            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')
                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]
                    if parameters['tagging_format'] == 'bioes':
                        split_line.pop()
                    gold_label_original = split_line[-1]
                    assert(token == token_original and gold_label == gold_label_original) 
                    break            
            split_line.append(prediction)
            output_string += ' '.join(split_line) + '\n'
        output_file.write(output_string+'\n')

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

#
#    np.save('SalmanTest/SalmanPred%s%s'%(i,dataset_type),predictions)
    #A.append(np.zeros(99))
    #C.append(np.zeros(99))
    np.save('SalmanTest/NAMEMr1mainSalmanUnary_scores%s'%dataset_type,A)
#    np.save('SalmanTest/SalmanUnary_scores%s%s'%(i,dataset_type),unary_scores)
 #   np.save('SalmanTest/myx3mainSalmanPred%s'%dataset_type,B)
    np.save('SalmanTest/NAMEMr1mainSalmanCCC%s'%dataset_type,C)
 #   np.save('SalmanTest/myx3ti',ti)
 #   np.save('SalmanTest/myx3chi',chi)
 #   np.save('SalmanTest/myx3tl',tl)
 #   np.save('SalmanTest/myx3lvi',lvi)
#
    output_file.close()
    original_conll_file.close()

    if dataset_type != 'deploy':
        if parameters['main_evaluation_mode'] == 'conll':
            conll_evaluation_script = os.path.join('.', 'conlleval')
            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath, conll_output_filepath)
            os.system(shell_command)
            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)
        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions, all_y_true, dataset, parameters['main_evaluation_mode'])
            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices, target_names=new_label_names))

    return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, stats_graph_folder, dataset_filepaths):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number, parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths


