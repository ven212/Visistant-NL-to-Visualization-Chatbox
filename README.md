# 📊 Visistant — NL-to-Visualization Chatbot

A faithful implementation of the **Visistant** paper architecture:
> "A Conversational Chatbot for Natural Language to Visualizations With Gemini Large Language Models"

---

## 🏗️ Architecture (from the paper)

```
User Query
    ↓
Initial Prompt Generation  (column names, types, top-20 categorical values)
    ↓
LangChain ConversationChain
    ├── ConversationBufferWindowMemory  (k=3 window)
    └── PromptTemplate  {history} + {input}
    ↓
Google Gemini Pro (temperature=0.1)
    ↓
Generated Plotly Code
    ↓
Code Extraction + Execution
    ↓
Plotly Chart rendered in Streamlit
    ↓
(Optional) Gemini Pro Vision → AI Insights
```

---

## ⚙️ Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a Google Gemini API Key

- Visit: https://makersuite.google.com/app/apikey
- Create a key (free tier available)

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. **Enter your API key** in the sidebar
2. **Upload a CSV file** — supports multiple files (stack-based selector)
3. **Choose Mode:**
   - **Default** — all columns sent to LLM
   - **Advanced** — select specific columns (reduces token length by ~75% per paper)
4. **Toggle "AI Insights"** — uses Gemini Vision to analyse your chart
5. **Type your query** in plain English and press Send

### Example Queries
```
Show total sales by region as a bar chart, sorted ascending
What is the trend of oil production since 2004?
Plot the correlation between engine size and horsepower
Give me a pie chart of product categories
Show the distribution of retail prices by car type
Change the previous bar chart to a line chart
```

---

## 📋 Key Implementation Details

| Feature | Implementation |
|---------|---------------|
| LLM | `gemini-pro` via `langchain-google-genai` |
| Memory | `ConversationBufferWindowMemory(k=3)` |
| Temperature | `0.1` (low — deterministic code) |
| Chart Library | Plotly Express + Graph Objects |
| Insights | `gemini-pro-vision` on chart image |
| Data Cleaning | Drop duplicates + median/mode imputation |
| Token Optimization | Advanced mode — only selected columns in prompt |
| Categorical Handling | Top-20 most frequent values per column |

---

## 📦 File Structure

```
visistant/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🔧 Troubleshooting

**"Gemini Vision not available"** — Some API tiers don't support `gemini-pro-vision`. Insights will be skipped gracefully.

**"Generated code had an error"** — Rephrase your query or try a simpler version first. The LLM occasionally misidentifies column names.

**Charts not rendering** — Make sure `kaleido` is installed (`pip install kaleido`) for image export needed by Gemini Vision.
