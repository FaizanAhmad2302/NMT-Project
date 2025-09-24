# Urdu-Roman Neural Machine Translation

A neural machine translation (NMT) system for converting Urdu text to Roman Urdu using sequence-to-sequence models with attention mechanisms.

## Overview

This project implements a character-level neural machine translation system that translates Urdu script to Roman Urdu (Romanized Urdu using Latin script). The system uses a bidirectional LSTM encoder-decoder architecture with Luong attention mechanism to achieve accurate transliteration.

## Features

- **Character-level translation**: Works at character level for better handling of morphology
- **Bidirectional LSTM Encoder**: Captures context from both directions
- **Luong Attention Mechanism**: Focuses on relevant parts of input during translation
- **Teacher Forcing**: Improves training stability
- **Early Stopping**: Prevents overfitting
- **Comprehensive Evaluation**: BLEU score, Character Error Rate (CER), and Perplexity metrics

## Project Structure

```
urdu-roman-nmt/
├── data/                    # Dataset files
│   ├── cleaned_final.csv   # Main dataset (Urdu, RomanUrdu pairs)
│   ├── splits.json         # Train/validation/test splits
│   ├── vocab.json          # Character vocabulary
│   ├── train.csv           # Training set
│   ├── val.csv             # Validation set
│   └── test.csv            # Test set
├── src/                     # Source code
│   ├── dataset.py          # Dataset handling and data loaders
│   ├── model.py            # Neural network architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── predict.py          # Inference script
│   └── vocab.py            # Vocabulary utilities
├── runs/                    # Training outputs
│   └── base/
│       ├── best.pt         # Best model checkpoint
│       └── config.json     # Training configuration
├── experiments/             # Experiment logs
├── make_split_csvs.py      # Data preprocessing script
├── urdu_input.txt          # Sample input for testing
└── requirements.txt        # Python dependencies
```

## Dataset

The dataset contains approximately 20,000+ parallel Urdu-Roman Urdu sentence pairs. The data is split into:
- **Training set**: 50% of the data
- **Validation set**: 25% of the data  
- **Test set**: 25% of the data

### Sample Data Format
```
Urdu: انکہ سے دور نہ ہو دل سے اتر جایے گا
RomanUrdu: aankh se duur na ho dil se utar jaaegaa
```

## Model Architecture

### Encoder
- **Embedding Layer**: Character embeddings (256 dimensions)
- **Bidirectional LSTM**: 2 layers, 512 hidden dimensions
- **Dropout**: 0.3 for regularization

### Decoder
- **Embedding Layer**: Character embeddings (256 dimensions)
- **LSTM**: 4 layers, 512 hidden dimensions
- **Luong Attention**: Computes attention weights between encoder outputs and decoder hidden states
- **Output Projection**: Linear layer for vocabulary prediction

### Special Tokens
- `<pad>` (ID: 0): Padding token
- `<sos>` (ID: 1): Start of sequence
- `<eos>` (ID: 2): End of sequence
- `<unk>` (ID: 3): Unknown character

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/FaizanAhmad2302/NMT-Project>
cd NMT-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Split the main dataset into train/validation/test sets:
```bash
python make_split_csvs.py
```

### 2. Training

Train the model with default parameters:
```bash
python src/train.py
```

Train with custom parameters:
```bash
python src/train.py --emb 256 --hid 512 --batch 64 --lr 5e-4 --epochs 20
```

#### Training Parameters
- `--emb`: Embedding dimension (default: 256)
- `--hid`: Hidden dimension (default: 512)
- `--enc_layers`: Number of encoder layers (default: 2)
- `--dec_layers`: Number of decoder layers (default: 4)
- `--dropout`: Dropout rate (default: 0.3)
- `--batch`: Batch size (default: 64)
- `--lr`: Learning rate (default: 5e-4)
- `--epochs`: Number of epochs (default: 20)
- `--patience`: Early stopping patience (default: 5)

### 3. Evaluation

Evaluate the trained model:
```bash
python src/evaluate.py --ckpt runs/base/best.pt
```

The evaluation provides:
- **Perplexity**: Measure of model uncertainty
- **BLEU Score**: Character-level BLEU score
- **Character Error Rate (CER)**: Edit distance-based metric

### 4. Inference

Translate Urdu text to Roman Urdu:
```bash
python src/predict.py --ckpt runs/base/best.pt --input_file urdu_input.txt
```

Or create your own input file with Urdu sentences (one per line) and specify the path.

## Example Usage

### Training Example
```bash
# Train with custom configuration
python src/train.py --emb 512 --hid 1024 --batch 32 --lr 1e-4 --epochs 30 --patience 7
```

### Prediction Example
Create a file `my_urdu.txt`:
```
وہ ایک دل جسے سب کچھ لٹا کے لوٹ لیا
ہوگا کویی ایسا بھی کہ غالبؔ کو نہ جانے
```

Run prediction:
```bash
python src/predict.py --ckpt runs/base/best.pt --input_file my_urdu.txt
```

Expected output:
```
Urdu: وہ ایک دل جسے سب کچھ لٹا کے لوٹ لیا
Roman Urdu: voh ek dil jise sab kuchh luta ke loot liya
----------------------------------------
Urdu: ہوگا کویی ایسا بھی کہ غالبؔ کو نہ جانے
Roman Urdu: hoga koi aisa bhi ke ghalib ko na jaane
----------------------------------------
```

## Model Performance

The model achieves competitive performance on the test set:
- **Perplexity**: Low perplexity indicates good model calibration
- **BLEU Score**: Character-level BLEU score for translation quality
- **Character Error Rate**: Edit distance-based accuracy metric

## Technical Details

### Training Features
- **Teacher Forcing**: Uses ground truth during training (ratio: 0.5)
- **Gradient Clipping**: Prevents exploding gradients (max norm: 1.0)
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Early Stopping**: Stops training when validation loss stops improving

### Inference Features
- **Greedy Decoding**: Deterministic decoding for consistent results
- **Dynamic Length**: Adapts output length based on input
- **Early Termination**: Stops when EOS token is generated

## Dependencies

- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **nltk**: Natural language processing utilities
- **tqdm**: Progress bars

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{urdu-roman-nmt,
  title={Urdu-Roman Neural Machine Translation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/urdu-roman-nmt}
}
```

## Acknowledgments

- The Urdu-Roman parallel corpus used for training
- PyTorch team for the excellent deep learning framework
- The open-source community for various tools and libraries
