# SignStream

SignStream is a real-time ASL translation system that combines a high-performance React web application with a Python-based AI research environment.

## Project Structure

### 🌐 frontend/
The "Restaurant". A React + Vite application serving the real-time experience.
- **Frontend**: React, TypeScript, Vite, Tailwind CSS.
- **AI Runtime**: MediaPipe (Hands) + ONNX Runtime Web (Inference).
- **Logic**: No backend required (Client-side AI).
- **Run**: `cd frontend && npm run dev`

### 🧪 ai-lab/
The "Kitchen". A Python environment with Jupyter Notebooks for data processing, model training, and fine-tuning.
- **notebooks/sign_language.ipynb**: Main notebook for training the sign recognition model.
- **notebooks/finetune_personal.ipynb**: Notebook for fine-tuning the model with personal data recorded from the frontend.
- **data/**: Storage for external (raw) and processed datasets.

## Getting Started

1. **Web Application**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **AI Lab (Python)**:
   It is recommended to use a virtual environment or conda.
   ```bash
   cd ai-lab
   pip install -r requirements.txt
   ```
   Then run Jupyter Labs or open the notebooks in your IDE:
   ```bash
   jupyter lab
   ```
   