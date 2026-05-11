from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import io
import re
import html


from transformers import AutoModel, AutoTokenizer

# congifuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label_mapping = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# arch model from development model step
    
class IndoBERT_BiLSTM(nn.Module):
    def __init__(
        self,
        bert_model_name=MODEL_NAME,
        hidden_dim=128,
        num_classes=3,
        dropout=0.25,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Mempertahankan menggunakan token (CLS & SEP)
        sequence_output = outputs.last_hidden_state  # (B, T, H)

        lengths = attention_mask.sum(dim=1)

        # Mengecualikan padding dari IndoBERT
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence_output,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.bilstm(packed)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)

        return self.classifier(hidden)


app = FastAPI(
    title="Sentiment Analysis API",
    description="Analisis Sentimen 17+8 Tuntutan Rakyat",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loat state_dict model
model_indobert = None
model_hybrid = None

@app.on_event("startup")
def load_models():

    global model_indobert, model_hybrid

    print("Loading models...")

    # model_indobert = IndoBERTClassifier(MODEL_NAME)
    # model_indobert.load_state_dict(
    #     torch.load("model_indobert_best.pth", map_location=DEVICE)
    # )
    # model_indobert.to(DEVICE)
    # model_indobert.eval()

    model_hybrid = IndoBERT_BiLSTM(MODEL_NAME)
    model_hybrid.load_state_dict(
        torch.load("indoBERT_BiLSTM_model05.pth", map_location=DEVICE)
    )
    model_hybrid.to(DEVICE)
    model_hybrid.eval()

    print("Models loaded successfully ✅")


# NOTE:
# BiLSTM-only butuh vocab_size.
# Jika Anda train dengan custom vocab, wajib load vocab.
# Untuk sekarang kita asumsikan menggunakan tokenizer BERT input.

# Preprocessing

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'@(\w+)', r'\1', text)
    text = re.sub(r'\brt\s+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'([!?\.])\1{3,}', r'\1\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F004-\U0001F0CF"
        "\U0001F000-\U0001F02F"
        "\U0001F1E0-\U0001F1FF"
        "\U0001F200-\U0001F251"
        "]",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    return text


# Interface Function

def predict_text(text: str, model_name: str):

    clean_text = preprocess_text(text)

    encoding = tokenizer(
        clean_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    if model_name == "IndoBERT":
        model = model_indobert
    else:
        model = model_hybrid

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)

        probabilities = probs.squeeze().cpu().tolist()
        pred_index = torch.argmax(probs, dim=1).item()

    return {
        "label": label_mapping[pred_index],
        "confidence": probabilities[pred_index],
        "probabilities": {
            label_mapping[i]: probabilities[i]
            for i in range(len(probabilities))
        }
    }


# API Endpoints

class TextInput(BaseModel):
    text: str
    model: str


@app.get("/")
def root():
    return {"status": "Backend Running 🚀"}


@app.post("/predict")
def predict(input: TextInput):

    result = predict_text(input.text, input.model)

    return {
        "raw_text": input.text,
        "clean_text": preprocess_text(input.text),
        "model_used": input.model,
        "prediction": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"]
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...), model: str = "IndoBERT-BiLSTM"):

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Validasi hanya 1 kolom
    if df.shape[1] != 1:
        return JSONResponse(
            status_code=400,
            content={"error": "CSV harus memiliki tepat 1 kolom teks."}
        )

    df.columns = ["text"]

    clean_texts = []
    predictions = []
    confidences = []

    for text in df["text"]:
        result = predict_text(text, model)

        clean_texts.append(preprocess_text(text))
        predictions.append(result["label"])
        confidences.append(round(result["confidence"], 4))

    output_df = pd.DataFrame({
        "Teks Preprocessing": clean_texts,
        "Sentimen": predictions,
        "Confidence": confidences
    })

    return output_df.to_dict(orient="records")

