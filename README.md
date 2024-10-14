
# Baby Attitude Classification

This project is a machine learning model that predicts a baby's attitude (happy or sad) based on an uploaded image, using TensorFlow and Keras. It uses a Convolutional Neural Network (CNN) for accurate classification.

## Features
- Upload an image of a baby
- Receive fast and accurate predictions of the baby's attitude (happy or sad)
- Simple web interface using Streamlit

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/Alekiie/Baby-Attitude.git
cd Baby-Attitude
```

### Step 2: Create a virtual environment
```bash
# For Linux/macOS
python3 -m venv venv

# For Windows
python -m venv venv
```

### Step 3: Activate the virtual environment
```bash
# For Linux/macOS
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

### Step 4: Install the required packages
```bash
pip install -r requirements.txt
```

### Step 5: Run the project
To start the Streamlit app and view the classification interface:
```bash
streamlit run page.py
```

The app will be running on http://localhost:8501.

## How It Works
1. Upload an image of a baby.
2. The model processes the image and predicts whether the baby is happy or sad.
3. Results are displayed in the Streamlit interface.

## License
This project is open-source and available under the [MIT License](LICENSE).
