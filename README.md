# NewsBot Intelligence System
## ITAI2373 Mid-Term Project

**Student:** Ahmet Burak Solak  
**Student Number:** W216974195

---

## Project Overview

The NewsBot Intelligence System is a comprehensive Natural Language Processing (NLP) application that automatically processes, categorizes, and extracts insights from BBC news articles. This system integrates all NLP techniques from Modules 1-8, including text preprocessing, feature extraction, sentiment analysis, named entity recognition, and multi-class classification.

### Key Features

- **Automated Text Classification**: Categorizes news articles into 5 categories (business, entertainment, politics, sport, tech)
- **Sentiment Analysis**: Determines emotional tone and polarity of articles
- **Named Entity Recognition**: Extracts people, organizations, locations, dates, and monetary values
- **TF-IDF Feature Extraction**: Identifies important terms distinguishing categories
- **POS Tagging & Syntax Analysis**: Analyzes grammatical patterns and writing styles
- **Multi-Algorithm Classification**: Compares Naive Bayes, Logistic Regression, SVM, and Random Forest

---

## Installation Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy English model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Download NLTK data** (will be downloaded automatically when running the notebook):
   - The notebook includes code to download required NLTK data (punkt, stopwords, wordnet, averaged_perceptron_tagger)

---

## Dataset

This project uses the **BBC News Train.csv** dataset containing:
- **ArticleId**: Unique identifier for each article
- **Text**: Full article text content
- **Category**: Article category (business, entertainment, politics, sport, tech)

The dataset includes over 1,400 articles distributed across 5 categories.

---

## Usage

### Running the Notebook

1. Open `NewsBot_System.ipynb` in:
   - **VS Code** with Python extension (recommended)
   - **Jupyter Notebook/Lab**
   - Any IDE with notebook support

2. Execute cells sequentially from top to bottom

3. The notebook is organized into 8 modules:
   - **Module 1**: Business Context and Application
   - **Module 2**: Text Preprocessing Pipeline
   - **Module 3**: TF-IDF Feature Extraction
   - **Module 4**: Part-of-Speech Pattern Analysis
   - **Module 5**: Syntax Parsing and Semantic Analysis
   - **Module 6**: Sentiment and Emotion Analysis
   - **Module 7**: Multi-Class Text Classification
   - **Module 8**: Named Entity Recognition

### Using the Unified Analysis Function

After running all cells, you can analyze any article using:

```python
result = analyze_article("Your article text here")
print(result['predicted_category'])
print(result['sentiment'])
print(result['entities'])
```

---

## Project Structure

```
ITAI2373-NewsBot-Midterm/
│
├── NewsBot_System.ipynb          # Main implementation notebook
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── BBC News Train.csv             # Training dataset
├── BBC News Test.csv              # Test dataset (optional)
├── BBC News Sample Solution.csv   # Test labels (optional)
└── venv/                          # Virtual environment (gitignored)
```

---

## Technical Implementation

### Module 1: Business Context
- Application description and use cases
- Target user identification
- Value proposition

### Module 2: Text Preprocessing
- Text cleaning and normalization
- Tokenization using spaCy
- Stop word removal
- Lemmatization

### Module 3: TF-IDF Analysis
- Vectorization with optimized parameters
- Top term extraction per category
- Feature importance visualization

### Module 4: POS Tagging
- Part-of-speech extraction
- Category-specific pattern analysis
- Writing style comparison

### Module 5: Syntax Parsing
- Dependency parsing
- Syntactic feature engineering
- Sentence complexity analysis

### Module 6: Sentiment Analysis
- Polarity and subjectivity scoring
- Sentiment distribution by category
- Emotional tone classification

### Module 7: Classification
- Multiple algorithm implementation
- Feature combination (TF-IDF + POS + Sentiment)
- Model evaluation and comparison
- Cross-validation

### Module 8: Named Entity Recognition
- Entity extraction (PERSON, ORG, GPE, DATE, MONEY)
- Entity frequency analysis
- Category-specific entity patterns

---

## Dependencies

All dependencies are listed in `requirements.txt`:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **spacy**: NLP and entity recognition
- **nltk**: Natural language toolkit
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization
- **textblob**: Sentiment analysis
- **wordcloud**: Word cloud generation
- **ipykernel**: Jupyter notebook support

---

## Results and Insights

### Classification Performance
The system achieves high accuracy in categorizing news articles across all 5 categories using a combination of TF-IDF features and linguistic features.

### Key Findings
1. Each category exhibits distinctive linguistic patterns
2. Sentiment varies significantly across categories
3. Entity types differ by category (e.g., sports articles mention more PERSON entities)
4. POS distributions reveal writing style differences

### Business Value
- Automates content categorization with high accuracy
- Extracts actionable insights (sentiment, entities, trends)
- Scales to process large volumes of articles
- Reduces manual content management effort

---

## Limitations

- Processing time increases with dataset size (some modules process samples for performance)
- Accuracy depends on training data quality
- Requires English language articles
- Computational resources needed for full dataset processing

---

## Future Enhancements

- Real-time processing capabilities
- Multi-language support
- Web interface/dashboard
- Advanced sentiment analysis models
- Custom entity training
- Topic modeling integration
- Trend analysis over time

---

## Contact

**Student:** Ahmet Burak Solak  
**Student Number:** W216974195  
**Course:** ITAI2373

---

## License

This project is created for educational purposes as part of the ITAI2373 course.


