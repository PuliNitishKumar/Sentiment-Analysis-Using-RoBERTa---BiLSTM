import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RoBERTa-BiLSTM Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RoBERTaBiLSTM(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, lstm_layers=2, dropout=0.3):
        super(RoBERTaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.hidden_dim = hidden_dim
        
        for param in self.roberta.parameters():
            param.requires_grad = True
        
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        lstm_output, (hidden, cell) = self.bilstm(sequence_output)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(hidden_concat)
        logits = self.fc(output)
        return logits

@st.cache_resource
def load_model_and_tokenizer(dataset='twitter'):
    """Load the trained model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    if dataset == 'sst2':
        model_path = 'models/sst2/roberta_bilstm_sst2_best.pth'
        num_classes = 2
    else:
        model_path = 'models/roberta_bilstm_best.pth'
        num_classes = 3
    
    if os.path.exists(model_path):
        model = RoBERTaBiLSTM(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, tokenizer, device, num_classes
    return None, tokenizer, device, num_classes

@st.cache_data
def load_metadata(dataset='sst2'):
    """Load dataset metadata"""
    if dataset == 'sst2':
        metadata_path = 'data/processed/sst2/metadata.json'
    else:
        metadata_path = 'data/processed/metadata.json'
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_results(dataset='twitter'):
    """Load training results"""
    if dataset == 'sst2':
        results_path = 'results/sst2/results.json'
    else:
        results_path = 'results/results.json'
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a single text"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    return prediction.item(), probabilities.cpu().numpy()[0]

def page_problem_statement():
    st.title("💬 Context-Aware Hybrid Model for Sentiment Analysis")
    st.markdown("### RoBERTa-BiLSTM Architecture")
    
    st.markdown("""
    ## 🎯 Problem Statement
    
    With the rapid advancement of technology, online activity has become integral to everyday life. 
    People express opinions, provide feedback, and share feelings across various platforms including 
    social media, education, business, entertainment, and sports.
    
    ### Challenges in Sentiment Analysis:
    - **Lexical Diversity**: Comments exhibit varied vocabulary and expressions
    - **Long Dependencies**: Text contains complex contextual relationships
    - **Unknown Symbols**: Presence of emojis, slang, and non-standard text
    - **Imbalanced Datasets**: Uneven distribution of sentiment classes
    - **Sequential Processing**: Traditional models process text sequentially (slower)
    
    ### Our Solution: RoBERTa-BiLSTM Hybrid Model
    
    This project implements a novel hybrid deep learning model that combines:
    - **RoBERTa**: Robustly Optimized BERT Pretraining Approach for meaningful word embeddings
    - **BiLSTM**: Bidirectional Long Short-Term Memory networks for capturing contextual semantics
    
    The hybrid approach leverages both Transformer-based parallel processing and sequential 
    context modeling to achieve superior performance.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    dataset = st.session_state.get('dataset', 'twitter')
    dataset_name = "Stanford SST-2" if dataset == 'sst2' else "Twitter US Airline"
    
    with col1:
        st.metric("Dataset", dataset_name, help="Selected dataset")
    
    with col2:
        results = load_results(dataset)
        if results:
            st.metric("Model Accuracy", f"{results['accuracy']*100:.2f}%", 
                     help="Test set accuracy")
    
    with col3:
        if results:
            st.metric("F1-Score", f"{results['f1_score']:.4f}", 
                     help="Weighted F1-score")
    
    st.markdown("""
    ## 🏗️ Model Architecture
    
    ```
    Input Text (Tweet)
        ↓
    RoBERTa Tokenizer (max_length=128)
        ↓
    RoBERTa Encoder (768-dim embeddings)
        ↓
    Bidirectional LSTM (256 hidden units × 2 layers)
        ↓
    Concatenate Forward & Backward Hidden States
        ↓
    Dropout (0.3)
        ↓
    Fully Connected Layer (3 classes)
        ↓
    Softmax (Positive, Negative, Neutral)
    ```
    
    ### Key Advantages:
    ✅ **Parallel Processing**: RoBERTa processes entire sequence simultaneously  
    ✅ **Contextual Understanding**: BiLSTM captures long-range dependencies  
    ✅ **Transfer Learning**: Pre-trained RoBERTa provides rich semantic representations  
    ✅ **Bidirectional Context**: Forward and backward LSTM capture full context  
    """)

def page_training_data():
    st.title("📊 Sample Training Data")
    
    dataset = st.session_state.get('dataset', 'twitter')
    metadata = load_metadata(dataset)
    
    if metadata:
        st.markdown(f"""
        ### Dataset: {metadata['dataset']}
        
        **Statistics:**
        - Total Samples: {metadata['total_samples']:,}
        - Training Samples: {metadata['train_samples']:,}
        - Test Samples: {metadata['test_samples']:,}
        - Number of Classes: {metadata['num_classes']}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Set Distribution")
            train_dist = pd.DataFrame.from_dict(
                metadata['train_distribution'], 
                orient='index', 
                columns=['Count']
            )
            st.dataframe(train_dist, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            train_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Training Set Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Test Set Distribution")
            test_dist = pd.DataFrame.from_dict(
                metadata['test_distribution'], 
                orient='index', 
                columns=['Count']
            )
            st.dataframe(test_dist, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            test_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Test Set Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("### Sample Tweets")
        train_data_path = 'data/processed/train_data.csv'
        if os.path.exists(train_data_path):
            df = pd.read_csv(train_data_path)
            sample_df = df.sample(min(10, len(df)))[['cleaned_text', 'sentiment']]
            st.dataframe(sample_df, use_container_width=True)
    else:
        st.warning("⚠️ No metadata found. Please run data_prep.py first.")

def page_preprocessing():
    st.title("🔧 Data Preprocessing")
    
    st.markdown("""
    ## Preprocessing Pipeline
    
    ### 1. Text Cleaning
    - Remove Twitter handles (@username)
    - Remove URLs and links
    - Remove hashtag symbols (keep text)
    - Remove special characters and punctuation
    - Convert to lowercase
    - Remove extra whitespace
    
    ### 2. Tokenization
    - **Tokenizer**: RoBERTa Tokenizer (Byte-Pair Encoding)
    - **Max Length**: 128 tokens
    - **Padding**: Pad sequences to max length
    - **Truncation**: Truncate longer sequences
    - **Special Tokens**: [CLS] and [SEP] added automatically
    
    ### 3. Label Encoding
    - Negative → 0
    - Neutral → 1  
    - Positive → 2
    
    ### 4. Data Splitting
    - **Train Set**: 80% (stratified)
    - **Test Set**: 20% (stratified)
    - Ensures balanced class distribution
    """)
    
    dataset = st.session_state.get('dataset', 'twitter')
    metadata = load_metadata(dataset)
    if metadata:
        st.markdown("### Preprocessing Metadata")
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                'Dataset': metadata['dataset'],
                'Total Samples': metadata['total_samples'],
                'Classes': metadata['classes']
            })
        
        with col2:
            st.json({
                'Label Mapping': metadata['label_mapping'],
                'Train Samples': metadata['train_samples'],
                'Test Samples': metadata['test_samples']
            })
        
        st.success("✅ Preprocessing completed successfully!")
    else:
        st.warning("⚠️ Run `python data_prep.py` to preprocess the data.")

def page_model_training():
    st.title("🎓 Model Training")
    
    dataset = st.session_state.get('dataset', 'twitter')
    num_classes = 2 if dataset == 'sst2' else 3
    optimization_info = ""
    
    if dataset == 'sst2':
        optimization_info = """
    ### ⚡ Optimizations Applied:
    - **Layer Freezing**: First 10/12 RoBERTa layers frozen
    - **Mixed Precision (FP16)**: Enabled for faster training
    - **Trainable Parameters**: 24% (30M out of 125M)
    - **Expected Training Time**: 35-60 minutes on RTX 4060
        """
    
    st.markdown(f"""
    ## RoBERTa-BiLSTM Architecture Details
    
    ### Model Components:
    
    **1. RoBERTa Encoder**
    - Pre-trained: `roberta-base`
    - Parameters: ~125M
    - Output dimension: 768
    - Fine-tuned during training
    
    **2. Bidirectional LSTM**
    - Hidden dimension: 256
    - Number of layers: 2
    - Bidirectional: Yes (forward + backward)
    - Dropout: 0.3 (between layers)
    - Total hidden output: 512 (256 × 2)
    
    **3. Classification Head**
    - Dropout: 0.3
    - Fully connected: 512 → {num_classes} classes
    - Activation: Softmax
    
    {optimization_info}
    
    ### Training Configuration:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Hyperparameters:**
        - Epochs: 5
        - Batch Size: 32
        - Learning Rate: 2e-5
        - Optimizer: AdamW
        - Scheduler: Linear warmup
        """)
    
    with col2:
        st.markdown("""
        **Loss & Metrics:**
        - Loss Function: CrossEntropyLoss
        - Metrics: Accuracy, Precision, Recall, F1
        - Gradient Clipping: Max norm 1.0
        - Device: GPU (if available)
        """)
    
    results = load_results(dataset)
    if results and 'training_history' in results:
        st.markdown("### Training History")
        
        history = results['training_history']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
            ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy')
            ax.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        training_time = results.get('training_time_minutes', 0)
        st.success(f"✅ Best Validation Accuracy: {results['best_val_accuracy']:.4f} | Training Time: {training_time:.1f} minutes")
    else:
        if dataset == 'sst2':
            st.warning("⚠️ Run `python train_model_optimized.py` to train the SST-2 model.")
        else:
            st.warning("⚠️ Run `python train_model.py` to train the Twitter model.")

def page_results():
    st.title("📈 Testing & Results")
    
    dataset = st.session_state.get('dataset', 'twitter')
    results = load_results(dataset)
    
    if results:
        dataset_name = results.get('dataset', 'Unknown')
        model_name = results.get('model', 'RoBERTa-BiLSTM')
        
        st.markdown(f"## Model Performance - {dataset_name}")
        st.markdown(f"**Model:** {model_name}")
        
        if dataset == 'sst2' and 'optimization' in results:
            opt = results['optimization']
            st.info(f"⚡ **Optimizations:** {opt.get('frozen_layers', 0)} layers frozen | Mixed Precision: {opt.get('mixed_precision', False)} | Trainable: {opt.get('trainable_params_percent', 0)}%")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{results['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{results['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{results['f1_score']:.4f}")
        
        if 'training_time_minutes' in results:
            st.metric("Training Time", f"{results['training_time_minutes']:.1f} minutes")
        
        st.markdown("### Confusion Matrix")
        if dataset == 'sst2':
            cm_path = 'results/sst2/figures/confusion_matrix.png'
        else:
            cm_path = 'results/figures/confusion_matrix.png'
            
        if os.path.exists(cm_path):
            img = Image.open(cm_path)
            st.image(img, use_container_width=True)
        else:
            st.warning(f"⚠️ Confusion matrix not found at {cm_path}")
        
        st.markdown("### Classification Report")
        if 'classification_report' in results:
            report_df = pd.DataFrame(results['classification_report']).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
        
        st.markdown("### Training History")
        if dataset == 'sst2':
            history_path = 'results/sst2/figures/training_history.png'
        else:
            history_path = 'results/figures/training_history.png'
            
        if os.path.exists(history_path):
            img = Image.open(history_path)
            st.image(img, use_container_width=True)
        else:
            st.warning(f"⚠️ Training history plot not found at {history_path}")
        
    else:
        if dataset == 'sst2':
            st.warning("⚠️ No SST-2 results found. Please train the model first using `python train_model_optimized.py`")
        else:
            st.warning("⚠️ No Twitter results found. Please train the model first using `python train_model.py`")

def page_prediction():
    st.title("🔮 Sentiment Prediction")
    
    dataset = st.session_state.get('dataset', 'twitter')
    model, tokenizer, device, num_classes = load_model_and_tokenizer(dataset)
    metadata = load_metadata(dataset)
    
    if model is None:
        st.error(f"❌ Model not found. Please train the model first by running `python train_model_optimized.py` (for SST-2) or `python train_model.py` (for Twitter)")
        return
    
    if metadata is None:
        st.error("❌ Metadata not found. Please run the data preprocessing script first.")
        return
    
    class_names = metadata['classes']
    # Handle both string and integer keys in label_mapping
    label_mapping = metadata['label_mapping']
    if dataset == 'sst2':
        # SST-2 has string keys "0", "1" -> convert to int keys
        label_to_sentiment = {int(k): v for k, v in label_mapping.items()}
    else:
        # Twitter has string keys as values
        label_to_sentiment = {v: k for k, v in label_mapping.items()}
    
    st.markdown("## Single Text Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        placeholder_text = "e.g., This movie is absolutely brilliant!" if dataset == 'sst2' else "e.g., The flight was delayed for 3 hours! Terrible service."
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder=placeholder_text,
            height=100
        )
    
    with col2:
        st.markdown("### Example Texts")
        if dataset == 'sst2':
            examples = [
                "This movie is absolutely brilliant!",
                "Terrible film, complete waste of time.",
                "An outstanding performance by the cast."
            ]
        else:
            examples = [
                "Great flight! Excellent service.",
                "Flight delayed again. Very disappointed.",
                "The flight was okay, nothing special."
            ]
        for ex in examples:
            if st.button(ex, key=ex):
                user_input = ex
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_sentiment(user_input, model, tokenizer, device)
                sentiment = label_to_sentiment[prediction]
                
                st.markdown("### Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                    color_map = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
                    
                    st.markdown(f"## {emoji_map.get(sentiment, '💬')} {sentiment.upper()}")
                    st.markdown(f"**Confidence:** {probabilities[prediction]*100:.2f}%")
                
                with col2:
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Sentiment': class_names,
                        'Probability': probabilities
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#FF6B6B' if s == 'negative' else '#4ECDC4' if s == 'neutral' else '#45B7D1' 
                             for s in class_names]
                    ax.barh(prob_df['Sentiment'], prob_df['Probability'], color=colors)
                    ax.set_xlabel('Probability')
                    ax.set_xlim(0, 1)
                    for i, v in enumerate(prob_df['Probability']):
                        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    st.markdown("---")
    st.markdown("## Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column",
        type=['csv'],
        help="CSV should have a column named 'text' containing tweets/reviews"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("❌ CSV must have a 'text' column")
        else:
            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Predict All", type="primary"):
                with st.spinner(f"Analyzing {len(df)} texts..."):
                    predictions = []
                    confidences = []
                    
                    for text in df['text']:
                        pred, probs = predict_sentiment(str(text), model, tokenizer, device)
                        predictions.append(label_to_sentiment[pred])
                        confidences.append(probs[pred])
                    
                    df['predicted_sentiment'] = predictions
                    df['confidence'] = confidences
                    
                    st.success("✅ Prediction complete!")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "sentiment_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    st.markdown("### Prediction Summary")
                    summary = df['predicted_sentiment'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    summary.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    ax.set_title('Predicted Sentiment Distribution')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()

def main():
    st.sidebar.title("📱 Navigation")
    
    # Dataset selector
    st.sidebar.markdown("### 🗂️ Select Dataset")
    dataset_options = {
        "Twitter US Airline": "twitter",
        "Stanford SST-2": "sst2"
    }
    
    dataset_display = st.sidebar.selectbox(
        "Choose Model/Dataset",
        list(dataset_options.keys()),
        help="Select which trained model to use"
    )
    dataset = dataset_options[dataset_display]
    
    # Store dataset in session state and clear cache when switching
    if 'dataset' not in st.session_state:
        st.session_state.dataset = dataset
    elif st.session_state.dataset != dataset:
        st.session_state.dataset = dataset
        st.cache_resource.clear()  # Clear model cache
        st.cache_data.clear()  # Clear data cache
        st.rerun()  # Force rerun to reload everything
    
    st.sidebar.markdown("---")
    
    pages = {
        "🎯 Problem Statement": page_problem_statement,
        "📊 Sample Training Data": page_training_data,
        "🔧 Data Preprocessing": page_preprocessing,
        "🎓 Model Training": page_model_training,
        "📈 Testing & Results": page_results,
        "🔮 Prediction": page_prediction
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 About
    **RoBERTa-BiLSTM Sentiment Analysis**
    
    A hybrid deep learning model combining:
    - RoBERTa (Transformer)
    - BiLSTM (Sequential)
    
    For context-aware sentiment classification.
    """)
    
    st.sidebar.markdown("---")
    
    # Show dataset-specific info
    if dataset == 'sst2':
        st.sidebar.markdown("**Dataset:** Stanford SST-2")
        st.sidebar.markdown("**Classes:** Positive, Negative")
        st.sidebar.markdown("**Optimization:** Layer Freezing (10/12)")
        st.sidebar.markdown("**Samples:** 67K train, 871 test")
    else:
        st.sidebar.markdown("**Dataset:** Twitter US Airline")
        st.sidebar.markdown("**Classes:** Positive, Negative, Neutral")
        st.sidebar.markdown("**Samples:** 11K train, 3K test")
    
    pages[selection]()

if __name__ == "__main__":
    main()
