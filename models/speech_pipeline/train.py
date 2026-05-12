import os
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,recall_score,f1_score,accuracy_score

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
METADATA_PATH=os.path.join(BASE_DIR,"metadata.csv")
PLOTS_DIR=os.path.join(BASE_DIR,"plots")
RESULTS_DIR=os.path.join(BASE_DIR,"results")
MODELS_DIR=os.path.join(BASE_DIR,"saved_models")
METRICS_DIR=os.path.join(BASE_DIR,"metrics")

os.makedirs(PLOTS_DIR,exist_ok=True)
os.makedirs(RESULTS_DIR,exist_ok=True)
os.makedirs(MODELS_DIR,exist_ok=True)
os.makedirs(METRICS_DIR,exist_ok=True)

SR=16000
DURATION=3
MAX_LEN=SR*DURATION
N_MELS=128
BATCH_SIZE=32
EPOCHS=30
LR=0.0003
PATIENCE=8
SEED=42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_TRAIN_SPEAKER="oaf"
TARGET_SPEAKER="yaf"
ADAPT_CLASSES=["anger","disgust","fear","happiness","neutral","sadness","surprise"]
ADAPT_RATIO=0.15

CLASS_NAMES=[
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

def preprocess_audio(path):
    audio,_=librosa.load(path,sr=SR)
    audio,_=librosa.effects.trim(audio,top_db=30)
    if len(audio)>MAX_LEN:
        audio=audio[:MAX_LEN]
    else:
        audio=np.pad(audio,(0,MAX_LEN-len(audio)))
    audio=librosa.util.normalize(audio)
    return audio.astype(np.float32)

def normalize_feature(x):
    return (x-x.mean())/(x.std()+1e-8)

def extract_features(audio):
    mel=librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )
    mel_db=librosa.power_to_db(mel,ref=np.max)
    delta1=librosa.feature.delta(mel_db)
    mfcc=librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=40,
        n_fft=1024,
        hop_length=256
    )
    mfcc=librosa.util.fix_length(
        mfcc,
        size=mel_db.shape[1],
        axis=1
    )
    mfcc=np.repeat(mfcc,4,axis=0)[:128]
    mel_db=normalize_feature(mel_db)
    delta1=normalize_feature(delta1)
    mfcc=normalize_feature(mfcc)
    stacked=np.stack([mel_db,delta1,mfcc],axis=0)
    return stacked.astype(np.float32)

def add_noise(audio):
    noise=np.random.randn(len(audio)).astype(np.float32)
    noise_level=np.random.uniform(0.002,0.008)
    return audio+(noise_level*noise)

def change_gain(audio):
    gain=np.random.uniform(0.75,1.25)
    return audio*gain

def time_shift(audio):
    shift=np.random.randint(-1600,1600)
    return np.roll(audio,shift)

def pitch_shift(audio):
    steps=np.random.uniform(-1.5,1.5)
    return librosa.effects.pitch_shift(y=audio,sr=SR,n_steps=steps)

def time_stretch(audio):
    rate=np.random.uniform(0.92,1.08)
    stretched=librosa.effects.time_stretch(y=audio,rate=rate)
    if len(stretched)>MAX_LEN:
        stretched=stretched[:MAX_LEN]
    else:
        stretched=np.pad(stretched,(0,MAX_LEN-len(stretched)))
    return stretched

def augment_audio(audio):
    aug=audio.copy()
    if random.random()<0.50:
        aug=add_noise(aug)
    if random.random()<0.50:
        aug=change_gain(aug)
    if random.random()<0.40:
        aug=time_shift(aug)
    if random.random()<0.35:
        aug=pitch_shift(aug)
    if random.random()<0.35:
        aug=time_stretch(aug)
    if len(aug)>MAX_LEN:
        aug=aug[:MAX_LEN]
    else:
        aug=np.pad(aug,(0,MAX_LEN-len(aug)))
    aug=librosa.util.normalize(aug)
    return aug.astype(np.float32)

class SERDataset(Dataset):
    def __init__(self,df,encoder,augment=False,use_raw_aug=False):
        self.df=df.reset_index(drop=True)
        self.encoder=encoder
        self.augment=augment
        self.use_raw_aug=use_raw_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        if self.augment and self.use_raw_aug:
            audio=preprocess_audio(row["file_path"])
            audio=augment_audio(audio)
            x=extract_features(audio)
        else:
            x=np.load(row["feature_path"])
        x=torch.tensor(x,dtype=torch.float32)
        y=self.encoder.transform([row["emotion"]])[0]
        return x,torch.tensor(y,dtype=torch.long)

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.shortcut=nn.Sequential()
        if in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.pool=nn.MaxPool2d(2)

    def forward(self,x):
        identity=self.shortcut(x)
        out=torch.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=out+identity
        out=torch.relu(out)
        out=self.pool(out)
        return out

class UnifiedSERModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.cnn=nn.Sequential(
            ResBlock(3,32),
            nn.Dropout2d(0.25),
            ResBlock(32,64),
            nn.Dropout2d(0.25),
            ResBlock(64,128),
            nn.Dropout2d(0.25)
        )
        with torch.no_grad():
            dummy=torch.zeros(1,3,128,188)
            out=self.cnn(dummy)
            _,c,f,t=out.shape
            lstm_input_size=c*f
        self.lstm=nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention=nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.classifier=nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        x=self.cnn(x)
        b,c,f,t=x.shape
        x=x.permute(0,3,1,2)
        x=x.reshape(b,t,c*f)
        lstm_out,_=self.lstm(x)
        attention_weights=torch.softmax(self.attention(lstm_out),dim=1)
        context=torch.sum(attention_weights*lstm_out,dim=1)
        output=self.classifier(context)
        return output

def build_training_data(df):
    df["speaker_id"]=df["speaker_id"].str.lower()
    df["emotion"]=df["emotion"].str.lower()

    base_train_df=df[df["speaker_id"]==BASE_TRAIN_SPEAKER].reset_index(drop=True)
    target_df=df[df["speaker_id"]==TARGET_SPEAKER].reset_index(drop=True)

    adapt_parts=[]
    remaining_parts=[]

    for emotion in sorted(target_df["emotion"].unique()):
        emotion_df=target_df[target_df["emotion"]==emotion].reset_index(drop=True)

        if emotion in ADAPT_CLASSES:
            adapt_df,remain_df=train_test_split(
                emotion_df,
                train_size=ADAPT_RATIO,
                random_state=SEED,
                shuffle=True
            )
            adapt_parts.append(adapt_df)
            remaining_parts.append(remain_df)
        else:
            remaining_parts.append(emotion_df)

    adapt_df=pd.concat(adapt_parts).reset_index(drop=True)
    remaining_target_df=pd.concat(remaining_parts).reset_index(drop=True)

    val_df,test_df=train_test_split(
        remaining_target_df,
        test_size=0.70,
        random_state=SEED,
        stratify=remaining_target_df["emotion"]
    )

    train_df=pd.concat([base_train_df,adapt_df],axis=0)
    train_df=train_df.sample(frac=1,random_state=SEED).reset_index(drop=True)

    val_df=val_df.reset_index(drop=True)
    test_df=test_df.reset_index(drop=True)

    train_df.to_csv(os.path.join(BASE_DIR,"train_split.csv"),index=False)
    val_df.to_csv(os.path.join(BASE_DIR,"val_split.csv"),index=False)
    test_df.to_csv(os.path.join(BASE_DIR,"test_split.csv"),index=False)

    print("\nTraining Data Strategy:")
    print("Base speaker + balanced target-speaker adaptation samples")

    print("\nBase Train Speaker:")
    print(BASE_TRAIN_SPEAKER)

    print("\nTarget Speaker:")
    print(TARGET_SPEAKER)

    print("\nTarget Classes Added To Training:")
    print(ADAPT_CLASSES)

    print("\nTarget Class Add Ratio:")
    print(ADAPT_RATIO)

    print("\nTrain Speaker Counts:")
    print(train_df["speaker_id"].value_counts())

    print("\nValidation Speaker Counts:")
    print(val_df["speaker_id"].value_counts())

    print("\nTest Speaker Counts:")
    print(test_df["speaker_id"].value_counts())

    print("\nTrain Emotion Counts:")
    print(train_df["emotion"].value_counts().sort_index())

    print("\nValidation Emotion Counts:")
    print(val_df["emotion"].value_counts().sort_index())

    print("\nTest Emotion Counts:")
    print(test_df["emotion"].value_counts().sort_index())

    return train_df,val_df,test_df

def evaluate(model,loader,device):
    model.eval()
    predictions=[]
    actuals=[]
    with torch.no_grad():
        for inputs,labels in loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs,1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    accuracy=accuracy_score(actuals,predictions)*100
    uar=recall_score(actuals,predictions,average="macro",zero_division=0)*100
    f1=f1_score(actuals,predictions,average="macro",zero_division=0)*100
    return accuracy,uar,f1,actuals,predictions

def save_curves(history):
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"],label="Train Loss")
    plt.plot(history["val_f1"],label="Validation Macro F1")
    plt.title("Training Loss and Validation Macro F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"training_curve.png"))
    plt.close()

def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("metadata.csv not found. Run preprocess.py first.")

    df=pd.read_csv(METADATA_PATH)

    encoder=LabelEncoder()
    encoder.fit(CLASS_NAMES)

    print("\nEmotion Classes:")
    print(encoder.classes_)

    print("\nTotal Samples:")
    print(len(df))

    print("\nClass Counts:")
    print(df["emotion"].value_counts().sort_index())

    train_df,val_df,test_df=build_training_data(df)

    train_dataset=SERDataset(train_df,encoder,augment=True,use_raw_aug=True)
    val_dataset=SERDataset(val_df,encoder,augment=False,use_raw_aug=False)
    test_dataset=SERDataset(test_df,encoder,augment=False,use_raw_aug=False)

    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    model=UnifiedSERModel(num_classes=len(encoder.classes_)).to(device)

    class_weights=torch.tensor(
        [1.1,1.0,1.0,1.15,1.0,1.0,1.05],
        dtype=torch.float32
    ).to(device)

    criterion=nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.03
    )

    optimizer=optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=5e-4
    )

    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    best_val_f1=0.0
    best_epoch=0
    counter=0

    history={
        "train_loss":[],
        "val_accuracy":[],
        "val_uar":[],
        "val_f1":[]
    }

    print("\nStarting Speech Emotion Recognition Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss=0.0

        for inputs,labels in tqdm(train_loader):
            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optimizer.step()

            running_loss+=loss.item()

        train_loss=running_loss/len(train_loader)

        val_acc,val_uar,val_f1,val_actuals,val_predictions=evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_uar"].append(val_uar)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}% | "
            f"Val UAR: {val_uar:.2f}% | "
            f"Val Macro F1: {val_f1:.2f}%"
        )

        if val_f1>best_val_f1:
            best_val_f1=val_f1
            best_epoch=epoch+1
            counter=0
            torch.save(model.state_dict(),os.path.join(MODELS_DIR,"best_model.pth"))
        else:
            counter+=1
            if counter>=PATIENCE:
                print("\nEarly stopping triggered")
                break

    print("\nTraining Complete")
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Validation Macro F1: {best_val_f1:.2f}%")

    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR,"best_model.pth"),
            map_location=device
        )
    )

    test_acc,test_uar,test_f1,test_actuals,test_predictions=evaluate(
        model,
        test_loader,
        device
    )

    print("\nFinal Test Results")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test UAR: {test_uar:.2f}%")
    print(f"Test Macro F1: {test_f1:.2f}%")

    report=classification_report(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    happiness_predictions=[]

    for actual,pred in zip(test_actuals,test_predictions):
        actual_label=encoder.inverse_transform([actual])[0]
        pred_label=encoder.inverse_transform([pred])[0]
        if actual_label=="happiness":
            happiness_predictions.append(pred_label)

    print("\nHAPPINESS PREDICTION ANALYSIS\n")
    print(Counter(happiness_predictions))

    report_dict=classification_report(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(RESULTS_DIR,"classification_report.csv")
    )

    cm=confusion_matrix(
        test_actuals,
        test_predictions,
        labels=list(range(len(CLASS_NAMES)))
    )

    cm_df=pd.DataFrame(cm,index=CLASS_NAMES,columns=CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR,"confusion_matrix.csv"))

    summary_df=pd.DataFrame([{
        "base_train_speaker":BASE_TRAIN_SPEAKER,
        "target_speaker":TARGET_SPEAKER,
        "target_classes_added_to_training":",".join(ADAPT_CLASSES),
        "target_class_add_ratio":ADAPT_RATIO,
        "test_accuracy":test_acc,
        "test_uar":test_uar,
        "test_macro_f1":test_f1,
        "happiness_recall":report_dict["happiness"]["recall"],
        "happiness_f1":report_dict["happiness"]["f1-score"],
        "anger_precision":report_dict["anger"]["precision"],
        "anger_recall":report_dict["anger"]["recall"]
    }])

    summary_df.to_csv(os.path.join(RESULTS_DIR,"summary.csv"),index=False)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,"confusion_matrix.png"))
    plt.close()

    metrics_df=pd.DataFrame({
        "Epoch":list(range(1,len(history["train_loss"])+1)),
        "Train_Loss":history["train_loss"],
        "Val_Accuracy":history["val_accuracy"],
        "Val_UAR":history["val_uar"],
        "Val_Macro_F1":history["val_f1"]
    })

    metrics_df.to_csv(os.path.join(METRICS_DIR,"training_metrics.csv"),index=False)

    save_curves(history)

    print("\nAll Outputs Saved Successfully")

if __name__=="__main__":
    main()