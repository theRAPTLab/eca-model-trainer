"""
    ECA - Training: Fine-tuning a Flan-T5-small language model 
    Date: 8/17/2023
    Author: Yeo Jin Kim (NC State University)
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset 
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score


# ----------------------------
# Training model
    
def preprocess_function(examples): 
    inputs = [doc for doc in examples[targetX]]
    model_inputs = tokenizer(inputs, max_length=maxlen, truncation=True, padding=True)

    # with tokenizer.as_target_tokenizer():  # current labels are not sentence but class labels. 
    labels = tokenizer(examples[targetY], max_length=maxlen, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def initData(trainFile, testFile):
    # load dataset 
    ds = load_dataset("csv", data_files={"train": "{}".format(trainFile), 
                                         "test": "{}".format(testFile)}) 
    # tokenized 
    token_ds= ds.map(preprocess_function, batched=True)
    
    print("\n* Data size: train ({}), test ({}) - train batch: {} (train step: {})".format(len(ds['train']), len(ds['test']), batch, 
                                                                                    np.ceil(len(ds['train'])/batch)*trainEpoch))    
    return ds, token_ds

def setModel(LANG_MODEL, device, tokenizer, modelPath, modelID, trainEpoch, lr, batch, warmup, save_steps,
             loadCheckpoint, train_data, eval_data):
    
    if not os.path.exists(modelPath): os.makedirs(modelPath)
    out_dir = modelPath+'{}'.format(modelID)
    
    if loadCheckpoint != None:
        model = AutoModelForSeq2SeqLM.from_pretrained(loadCheckpoint,local_files_only=True).to(device) 
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(LANG_MODEL)
        model = model.to(device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    early_stop = EarlyStoppingCallback(patience, 0.0)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = out_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate = lr,
        per_device_train_batch_size = batch,
        per_device_eval_batch_size = batch,
        weight_decay = 0.01,
        save_total_limit = 1,
        num_train_epochs = trainEpoch, 
        fp16 = False,  # only available with CUDA, otherwise set False
        warmup_steps=warmup,
        #overwrite_output_dir=True,
        #seed=seed,
        save_steps=save_steps,
        load_best_model_at_end = True
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_data, 
        eval_dataset = eval_data, 
        tokenizer = tokenizer,
        data_collator = data_collator,
        callbacks=[early_stop]        
    )
    
    return model, trainer

def train(trainer, modelPath, modelID, trainEpoch):    
        
    if trainEpoch>0: # For training
        # ---------------------------
        result = trainer.train()  
        # ---------------------------        
        print(result)
                
    else: # For testing only 
        result = None
    return result
 
    
# ----------------------------
# Test & Evaluation
    
# measure by class    
def getf1(rdf, target_label):
    tp = len(rdf[(rdf.label==target_label)&(rdf.pred==target_label)])
    fp = len(rdf[(rdf.label!=target_label)&(rdf.pred==target_label)])
    fn = len(rdf[(rdf.label==target_label)&(rdf.pred!=target_label)])
    f1 = tp / (tp + 1/2*(fp+fn))
    return f1, tp, fp, fn

def getAcc(rdf, target_label):
    tp = len(rdf[(rdf.label==target_label)&(rdf.pred==target_label)])
    total = len(rdf[rdf.label==target_label])
    acc = tp/total
    return acc, total

def getMCC(rdf, target_label):
    tp = len(rdf[(rdf.label==target_label)&(rdf.pred==target_label)])
    fp = len(rdf[(rdf.label!=target_label)&(rdf.pred==target_label)])
    fn = len(rdf[(rdf.label==target_label)&(rdf.pred!=target_label)])
    tn = len(rdf[(rdf.label!=target_label)&(rdf.pred!=target_label)])
    mcc = (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return mcc, tp, fp, fn, tn
            
    
def test_batch(device, tokenizer, model, ds, groupIDName, maxlen, testFile, logFile, test_batch_size=32):
    print("\n** Evaluation: ")
    rdf = pd.DataFrame(columns = [groupIDName, 'label', 'pred'])

    test_size = len(ds['test'])
    batch_num = int(np.ceil(test_size/test_batch_size))
    print("test: {}, batch size: {}, batch_num: {}, groupIDName: {}".format(ds['test'].num_rows, test_batch_size, batch_num, groupIDName))
    groupIDs = pd.read_csv(testFile, usecols = [groupIDName], header=0)
    
    for batchID in range(batch_num):
        idx = batchID*test_batch_size
        test_batch = ds['test'][idx:idx+test_batch_size]
        
        input_ids = tokenizer(test_batch[targetX], return_tensors="pt", padding=True).to(device).input_ids
        
        outputs = model.generate(input_ids, max_length = maxlen, do_sample=False)

        batch_res = pd.DataFrame(columns = [groupIDName, 'label', 'pred'])         
        #print("test_batch: {} batch_res_groupID: {}".format(len(test_batch[targetX]), len(groupIDs.loc[idx:idx+test_batch_size-1, groupIDName] )))
        
        batch_res[groupIDName] = groupIDs.loc[idx:idx+test_batch_size-1, groupIDName]        
        batch_res['label'] = test_batch[targetY]
        batch_res['pred'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        rdf = pd.concat([rdf, batch_res], axis=0)

    rdf.reset_index(drop=True, inplace=True)
    rdf.to_csv('{}'.format(logFile), index=False)
    return rdf      

def evaluate(rdf, ds, targetX, targetY, groupIDName, testFile, evalFile, evalByClass):
    print("\n** Evaluation:")
    scoredf = pd.DataFrame(columns = [groupIDName, 'mcc', 'f1', 'acc', 'tp','tn', 'fp', 'fn', 'total'])

    # measure by group 
    if evalByClass: # Not used due to response generaton
        for tlabel in sorted(groupIDName):
            f1, tp, fp, fn = getf1(rdf, tlabel)
            acc, total = getAcc(rdf, tlabel)
            mcc, tp, fp, fn, tn = getMCC(rdf, tlabel)
            scoredf.loc[len(scoredf)] = [tlabel, mcc, f1, acc, tp, tn, fp, fn, total]
        
    # measure across all the classes
    topic_res = rdf[rdf.label.isin(rdf.topic.unique())]
    print("topic_res: {}, rdf: {}".format(len(topic_res), len(rdf)))
    y_true = topic_res.label.values.tolist()
    y_pred = topic_res.pred.values.tolist()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_acc = accuracy_score(y_true, y_pred)
    macro_mcc = matthews_corrcoef(y_true, y_pred)

    scoredf.loc[len(scoredf)] = ['macro', macro_mcc, macro_f1, macro_acc] + np.sum(
        scoredf.loc[:, ['tp', 'tn','fp', 'fn', 'total']]).tolist()
    print("* Topic classification:")
    print(scoredf)
    
    scoredf.to_csv('{}'.format(evalFile), index=False)
    return scoredf


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, default='ci', help='model name')    
    parser.add_argument('-dataFile', type=str, default='eca_data', help='training file name (*.csv)')    
    parser.add_argument('-trainFile', type=str, default='eca_train', help='training file name (*.csv)')
    parser.add_argument('-testFile', type=str, default='eca_val', help='test file name (*.csv)')    
    parser.add_argument('-patience', type=int, default=3)
    parser.add_argument('-pretrain', type=str, default=None, help='checkpoint path of a pretraining model')
    parser.add_argument('-pretrainEpoch', type=int, default=0, help='training epochs of a pretraining model')
    parser.add_argument('-lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-warmup', type=int, default=100, help='warmup size')
    parser.add_argument('-save_steps', type=int, default=5000, help='step for saving model')
    parser.add_argument('-maxlen', type=int, default=256, help='max token length for input text')
    parser.add_argument('-targetX', type=str, default='sentence', help='target column for input')
    parser.add_argument('-targetY', type=str, default='labels', help='target column for output')
    parser.add_argument('-epoch', type=int,default=30, help='training epochs')    
    parser.add_argument('-evalByClass', type=int, default=0, help='add the evaluation measurements by class')
    args = parser.parse_args()  
    print(args)
    return args    


if __name__ == '__main__':
    args = getArgs()
    trainEpoch = args.epoch 
    modelName = args.model
    pretrain = args.pretrain
    pretrainEpoch = args.pretrainEpoch
    patience = args.patience
    lr = args.lr
    batch = args.batch
    warmup = args.warmup
    maxlen = args.maxlen
    save_steps = args.save_steps
    
    targetX = args.targetX
    targetY = args.targetY
    evalByClass = args.evalByClass
    
    
    totStartTime = time.time()
    
    LANG_MODEL = "google/flan-t5-small"
    # LANG_MODEL = "google/flan-t5-base"
    # LANG_MODEL = "t5-small"
    groupIDName = 'topic'
    model = None
    modelPath = 'model/' 
    dataFile = 'data/{}.csv'.format(args.dataFile)
    trainFile = 'data/{}.csv'.format(args.trainFile)
    testFile = 'data/{}.csv'.format(args.testFile)
    logFile = 'result/res_log.csv'
    evalFile = 'result/eval.csv'
    if not os.path.exists('result'): os.mkdir('result')
    modelID = "model_{}_lr{}_b{}_e{}".format(modelName, lr, batch, trainEpoch)
    
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL) 
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    
    # ---------------------------------------------
    # preprocessing
    if True:
        
        # iu = pd.read_csv("data/IU_ECA_Questions_and_Answers.csv", header=0)
        iu = pd.read_csv("data/Context_Field_Test.csv", header=0)
        cols= ['topic', 'context', 'sentence', 'labels']
        iu.columns = cols
        iu['sentence'] = "Type: " + iu.topic + ", context: " + iu.context + ", response: " + iu.sentence        
        alldata = iu[cols]
        print(f"IU data: response ({len(iu)})")
        print(f"All data: {len(alldata)}")
        alldata.to_csv("data/eca_data.csv", index=False)
        
        groupIDs = alldata[groupIDName].unique().tolist()
        print("groups: ", groupIDs)        
        
        # 2) Train / test split
        traindf, valdf = train_test_split(alldata, test_size=0.1, shuffle=True, random_state=0, stratify=alldata.topic)
        traindf.to_csv("data/eca_train.csv", index=False)
        valdf.to_csv("data/eca_val.csv", index=False)
        print("train data:\n{}".format(traindf.sample(frac=1).head(5)))
        print("test data:\n{}".format(valdf.sample(frac=1).head(5))) 
    # --------------------------------------------
    # Training and validation to decide the optimal training epoch (or other parameters)

    if False: # skip the hyperparmeter search
        print("\n** Hyperparameter search")
        if trainEpoch > 0:
            ds, token_ds = initData(trainFile, testFile) 
            model, trainer = setModel(LANG_MODEL, device, tokenizer, modelPath, modelID, trainEpoch, lr, batch, warmup, save_steps,
                                      loadCheckpoint=pretrain, train_data=token_ds["train"], eval_data = token_ds["test"])    

            result = train(trainer, modelPath, modelID, trainEpoch) # if trainEpoch==0, skip training and only test
            trainEpoch = result.metrics['epoch'] - patience # select the best training epoch (since we don't have validataion data for the final model
            bestModelPath = trainer.state.best_model_checkpoint
            print("\n ** Best epoch: {}, model: {}".format(trainEpoch, bestModelPath))
        
    # --------------------------------------------    
    # Final model training : training and test with all the data (due to lack of data) --> properly split train/ test when having enough data        
    print("\n** Final model training")
    ds, token_ds = initData(dataFile, dataFile) 
    
    # trainer.train_dataset = token_ds["train"]
    # trainer.eval_dataset = token_ds["test"]
    
    model, trainer = setModel(LANG_MODEL, device, tokenizer, modelPath, modelID, trainEpoch, lr, batch, warmup, save_steps,
                              loadCheckpoint=pretrain, train_data=token_ds["train"], eval_data = token_ds["test"])     
    result = train(trainer, modelPath, modelID, trainEpoch) # if trainEpoch==0, skip training and only test
    bestModelPath = trainer.state.best_model_checkpoint
    print("\n ** Best model: {}".format(bestModelPath))    
    
    # --------------------------------------------    
    # Test
    rdf = test_batch(device, tokenizer, model, ds, groupIDName, maxlen, dataFile, logFile, test_batch_size=batch)
    scoredf = evaluate(rdf, ds, targetX, targetY, groupIDName, dataFile, evalFile, evalByClass)      
    totLearnTime = (time.time() - totStartTime)/60 
    print("Total Time: {:.1f} min ({:.1f} hours)".format(totLearnTime, totLearnTime/60))

