import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

MODEL_PATH=os.path.join(BASE_DIR,"saved_models","best_model.pth")
TEST_CSV=os.path.join(BASE_DIR,"test_split.csv")
OUTPUT_CSV=os.path.join(BASE_DIR,"prediction_results.csv")
REPORT_CSV=os.path.join(BASE_DIR,"test_classification_report.csv")
CONFUSION_CSV=os.path.join(BASE_DIR,"test_confusion_matrix.csv")

CLASS_NAMES=[
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv1=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2=nn.BatchNorm2d(out_channels)

        self.shortcut=nn.Sequential()

        if in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                ),
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

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test split not found: {TEST_CSV}")

    df=pd.read_csv(TEST_CSV)

    encoder=LabelEncoder()
    encoder.fit(CLASS_NAMES)

    print("\nEmotion Classes:")
    print(encoder.classes_)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    model=UnifiedSERModel(
        num_classes=len(encoder.classes_)
    ).to(device)

    state_dict=torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("\nModel Loaded Successfully")

    results=[]
    actuals=[]
    predictions=[]

    with torch.no_grad():
        for _,row in tqdm(df.iterrows(),total=len(df)):
            feature_path=row["feature_path"]
            actual_emotion=row["emotion"]

            try:
                x=np.load(feature_path)

                if x.shape!=(3,128,188):
                    raise ValueError(f"Invalid feature shape: {x.shape}")

                x=torch.tensor(
                    x,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                output=model(x)

                probabilities=torch.softmax(
                    output,
                    dim=1
                )[0]

                predicted_idx=torch.argmax(
                    probabilities
                ).item()

                confidence=probabilities[predicted_idx].item()

                predicted_emotion=encoder.inverse_transform(
                    [predicted_idx]
                )[0]

                actuals.append(actual_emotion)
                predictions.append(predicted_emotion)

                results.append({
                    "file_name":os.path.basename(feature_path),
                    "file_path":feature_path,
                    "actual":actual_emotion,
                    "predicted":predicted_emotion,
                    "confidence":confidence,
                    "correct":actual_emotion==predicted_emotion
                })

            except Exception as e:
                results.append({
                    "file_name":os.path.basename(feature_path),
                    "file_path":feature_path,
                    "actual":actual_emotion,
                    "predicted":"ERROR",
                    "confidence":0.0,
                    "correct":False,
                    "error":str(e)
                })

                print(f"\nError processing {feature_path}")
                print(e)

    accuracy=accuracy_score(
        actuals,
        predictions
    )*100

    print(f"\nFinal Accuracy: {accuracy:.2f}%")

    report=classification_report(
        actuals,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    results_df=pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    report_dict=classification_report(
        actuals,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        REPORT_CSV
    )

    cm=confusion_matrix(
        actuals,
        predictions,
        labels=CLASS_NAMES
    )

    cm_df=pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(CONFUSION_CSV)

    print(f"\nPrediction CSV Saved To:\n{OUTPUT_CSV}")
    print(f"\nClassification Report CSV Saved To:\n{REPORT_CSV}")
    print(f"\nConfusion Matrix CSV Saved To:\n{CONFUSION_CSV}")

    wrong_df=results_df[results_df["correct"]==False]

    print(f"\nWrong Predictions: {len(wrong_df)}")

    if len(wrong_df)>0:
        print("\nWrong Prediction Summary:")
        print(
            wrong_df.groupby(
                ["actual","predicted"]
            ).size().sort_values(ascending=False)
        )

if __name__=="__main__":
    main()