import pandas as pd
import os

train = pd.read_csv('/usr/users/multimodalemotion/tdujardin/MELD.Raw/train_sent_emo.csv')
dev = pd.read_csv('/usr/users/multimodalemotion/tdujardin/MELD.Raw/dev_sent_emo.csv')

for index, row in train.iterrows():
    if not os.path.exists('/usr/users/multimodalemotion/tdujardin/MELD.Raw/train/dia' + str(row['Dialogue_ID']) + '_utt' + str(row['Utterance_ID'])):
        train.drop(index, inplace=True)
        print('dia' + str(row['Dialogue_ID']) + '_utt' + str(row['Utterance_ID']))

for index, row in dev.iterrows():
    if not os.path.exists('/usr/users/multimodalemotion/tdujardin/MELD.Raw/dev/dia' + str(row['Dialogue_ID']) + '_utt' + str(row['Utterance_ID'])):
        dev.drop(index, inplace=True)
        print('dia' + str(row['Dialogue_ID']) + '_utt' + str(row['Utterance_ID']))

#train.reset_index(inplace=True)
#dev.reset_index(inplace=True)
train.to_csv('/usr/users/multimodalemotion/tdujardin/MELD.Raw/train_sent_emo2.csv')
dev.to_csv('/usr/users/multimodalemotion/tdujardin/MELD.Raw/dev_sent_emo2.csv')