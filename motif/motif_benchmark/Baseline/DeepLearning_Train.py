import os
from tqdm import tqdm
import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet

import Utils.Metrics
import Utils.Threshold


def set_seed(args):
    np.random.seed(args)
    torch.manual_seed(args)


'''
    Initialization as follows:
'''
DataSetName = 'ChIP-seq'
NeuralNetworkNameList = ['DeepSNR', 'D_AEDNet']

use_gpu = torch.cuda.is_available()
print(use_gpu)
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''
for NeuralNetworkName in NeuralNetworkNameList:
    result = open('Result_' + NeuralNetworkName + '_1.txt', 'a')
    # result.write('TF_name\tAcc\tPre\tRec\tF1-S\tAUC\tAUPR\n')

    for TF in os.listdir('../Dataset/ChIP-seq'):
        TFsName = TF
        weight_path = 'Weight/' + NeuralNetworkName + '/' + TFsName + '_1.pth'
        if os.path.exists(weight_path):
            # print('Weight already exists, skip training for ' + TFsName)
            continue
        print(TFsName)
        bestAUPR = 0
        acc = 0
        pre = 0
        rec = 0
        f1 = 0
        auc = 0
        aupr = 0
        for seed in range(1, 10):
            set_seed(seed)
            FeatureMatrix_train, DenseLabel_train = Dataset.DataReader.DataReader(
                TFName=TFsName,
                DataSetName=DataSetName,
                type='train')
            FeatureMatrix_train, DenseLabel_train = np.array(FeatureMatrix_train), np.array(DenseLabel_train)
            FeatureMatrix_test, DenseLabel_test = Dataset.DataReader.DataReader(
                TFName=TFsName,
                DataSetName=DataSetName,
                type='test')
            FeatureMatrix_test, DenseLabel_test = np.array(FeatureMatrix_test), np.array(DenseLabel_test)

            ThresholdValue = 0.5

            LossFunction = nn.BCELoss().cuda()

            MaxEpoch = 10

            if NeuralNetworkName == 'DeepSNR':
                NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15).cuda()
            else:
                NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100).cuda()

            optimizer = torch.optim.AdamW(NeuralNetwork.parameters(), lr=5e-4, weight_decay=1e-5)

            TrainFeatureMatrix = torch.tensor(FeatureMatrix_train, dtype=torch.float32).unsqueeze(dim=1)
            TrainDenseLabels = torch.tensor(DenseLabel_train)
            TestFeatureMatrix = torch.tensor(FeatureMatrix_test, dtype=torch.float32).unsqueeze(dim=1)
            TestDenseLabels = torch.tensor(DenseLabel_test)
            TestLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TestFeatureMatrix, DenseLabel=TestDenseLabels,
                                                         BatchSize=32)

            for Epoch in range(MaxEpoch):
                # train
                perm = torch.randperm(len(TrainFeatureMatrix))
                TrainFeatureMatrix = TrainFeatureMatrix[perm]
                TrainDenseLabels = TrainDenseLabels[perm]
                TrainLoader = Dataset.DataLoader.SampleLoader(
                    FeatureMatrix=TrainFeatureMatrix,
                    DenseLabel=TrainDenseLabels,
                    BatchSize=32
                )
                NeuralNetwork.train()
                TrainProgressBar = tqdm(TrainLoader, colour='blue', desc=f'Training Epoch {Epoch}')
                running_loss = 0.0
                for i, data in enumerate(TrainProgressBar):
                    optimizer.zero_grad()
                    X, Y = data
                    X = X.cuda()
                    Y = Y.cuda()
                    Prediction = NeuralNetwork(X)
                    Loss = LossFunction(Prediction.squeeze(), Y.to(torch.float32))
                    Loss.backward()
                    optimizer.step()

                    running_loss += Loss.item()
                    if (i + 1) % 10 == 0 or (i + 1) == len(TrainLoader):
                        avg_loss = running_loss / (i + 1)
                        TrainProgressBar.set_postfix(loss=f'{avg_loss:.4f}')

                # test
                NeuralNetwork.eval()
                ValidProgressBar = tqdm(TestLoader, colour='green', desc=f'Testing Epoch {Epoch}')
                pred = np.array([])
                label = np.array([])
                logits = np.array([])
                for data in ValidProgressBar:
                    X, Y = data
                    X = X.cuda()
                    Y = Y.cuda()
                    Logits = NeuralNetwork(X)
                    Prediction = Utils.Threshold.Threshold(YPredicted=Logits.cpu(), ThresholdValue=ThresholdValue)
                    logits = np.append(logits, Logits.cpu().detach().numpy())
                    pred = np.append(pred, Prediction)
                    label = np.append(label, Y.cpu())

                Performance = np.zeros(shape=6, dtype=np.float32)
                Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5] = \
                    Utils.Metrics.EvaluationMetricsSequence(y_pred=pred, y_true=label, y_logits=logits)

                print(
                    f"Epoch {Epoch} - Acc={Performance[0]:.3f}, Pre={Performance[1]:.3f}, Rec={Performance[2]:.3f}, "
                    f"F1-S={Performance[3]:.3f}, AUC={Performance[4]:.3f}, AUPR={Performance[5]:.3f}")

                if Performance[5] > aupr:
                    acc = Performance[0]
                    pre = Performance[1]
                    rec = Performance[2]
                    f1 = Performance[3]
                    auc = Performance[4]
                    aupr = Performance[5]
                    os.makedirs('Weight/' + NeuralNetworkName, exist_ok=True)
                    torch.save(NeuralNetwork.state_dict(), 'Weight/' + NeuralNetworkName + '/' + TFsName + '_1.pth')
                else:
                    break
            if pre > 0.1 and rec > 0.1:
                break
        print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f,' % (
            acc, pre, rec, f1, auc, aupr))
        result.write(
            TFsName + '\t' + str(acc) + '\t' + str(pre) + '\t' + str(rec) + '\t' + str(f1) + '\t' + str(
                auc) + '\t' + str(
                aupr) + '\n')
        result.flush()
