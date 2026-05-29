import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

class BeritaCheckInferenceEngine:
    def __init__(self, bias_dir: str, sentiment_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Labels exactly matching the notebook configurations
        self.bias_labels = {0: "Non-Political/Objective", 1: "Political-Framed"}
        self.sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        # Load BERT Bias Detector
        print(f"Loading BERT Bias Model from: {bias_dir}")
        self.bias_tokenizer = BertTokenizer.from_pretrained(bias_dir)
        self.bias_model = BertForSequenceClassification.from_pretrained(bias_dir).to(self.device)
        self.bias_model.eval()
        
        # Load RoBERTa Sentiment Engine
        print(f"Loading RoBERTa Sentiment Model from: {sentiment_dir}")
        self.sentiment_tokenizer = RobertaTokenizer.from_pretrained(sentiment_dir)
        self.sentiment_model = RobertaForSequenceClassification.from_pretrained(sentiment_dir).to(self.device)
        self.sentiment_model.eval()

    @torch.no_grad()
    def infer(self, text: str, max_length: int = 256) -> dict:
        if not text.strip():
            return {"error": "Text segment is empty"}

        # 1. Evaluate Bias Model (BERT Binary)
        bias_enc = self.bias_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        bias_logits = self.bias_model(**bias_enc).logits
        bias_probs = F.softmax(bias_logits, dim=1).cpu().numpy()[0]
        bias_idx = int(torch.argmax(bias_logits, dim=1).item())

        # 2. Evaluate Sentiment Model (RoBERTa Multi-Class)
        sent_enc = self.sentiment_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        sent_logits = self.sentiment_model(**sent_enc).logits
        sent_probs = F.softmax(sent_logits, dim=1).cpu().numpy()[0]
        sent_idx = int(torch.argmax(sent_logits, dim=1).item())

        return {
            "bias_analysis": {
                "prediction_index": bias_idx,
                "label": self.bias_labels[bias_idx],
                "confidence": float(bias_probs[bias_idx]),
                "distribution": {self.bias_labels[k]: float(v) for k, v in enumerate(bias_probs)}
            },
            "sentiment_analysis": {
                "prediction_index": sent_idx,
                "label": self.sentiment_labels[sent_idx],
                "confidence": float(sent_probs[sent_idx]),
                "distribution": {self.sentiment_labels[k]: float(v) for k, v in enumerate(sent_probs)}
            }
        }