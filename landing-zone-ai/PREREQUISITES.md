# Prerequisites & Setup Guide

This guide details how to set up and run the `landing-zone-ai` system **without Docker** and **without a GPU**.

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Modern multi-core processor (Intel i5/i7, AMD Ryzen 5/7)
- **Disk Space**: ~2GB for code and dependencies + space for data

## Required Software

1.  **Node.js (v16 or higher)**
    *   Download: [https://nodejs.org/](https://nodejs.org/)
    *   Verify: `node -v`, `npm -v`

2.  **Python (v3.8 or higher)**
    *   Download: [https://www.python.org/](https://www.python.org/)
    *   Verify: `python --version`

3.  **MongoDB (Community Edition)**
    *   Download: [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
    *   Installation: Follow the instructions for your OS.
    *   Verify: `mongod --version`

## Installation Steps

### 1. Database Setup
*   Start the MongoDB service.
    *   **Windows**: Usually starts automatically as a service, or run `"C:\Program Files\MongoDB\Server\X.X\bin\mongod.exe"`
    *   **Mac/Linux**: `sudo systemctl start mongod` or `mongod`

### 2. Backend (Server) Setup
1.  Open a terminal and navigate to the `server` directory:
    ```bash
    cd landing-zone-ai/server
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Create/Verify `.env` file:
    *   Ensure a file named `.env` exists with the following content:
        ```
        PORT=3000
        MONGO_URI=mongodb://localhost:27017/landing-zone-ai
        PYTHON_SERVICE_URL=http://localhost:5000
        ```
4.  Start the server:
    ```bash
    npm run dev
    ```
    *   You should see: `Server running on port 3000` and `MongoDB Connected`.

### 3. AI Microservice (Python) Setup

#### Option A: Using Anaconda (Recommended)
1.  Open your Anaconda Prompt or Terminal.
2.  Navigate to the `python-ai` directory:
    ```bash
    cd landing-zone-ai/python-ai
    ```
3.  Create a new Conda environment:
    ```bash
    conda create -n landing-zone-ai python=3.9
    conda activate landing-zone-ai
    ```
4.  Install dependencies:
    ```bash
    # Install PyTorch (CPU version)
    conda install pytorch torchvision cpuonly -c pytorch
    
    # Install other requirements via pip
    pip install -r requirements.txt
    ```

#### Option B: Standard Python (venv)
1.  Open a **new** terminal and navigate to the `python-ai` directory:
    ```bash
    cd landing-zone-ai/python-ai
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
3.  Install dependencies:
    *   **Note:** We are installing CPU-only versions of PyTorch to avoid GPU requirements.
    ```bash
    pip install -r requirements.txt
    ```
    *   *If you encounter issues with `torch`, try installing the CPU version specifically:*
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ```
4.  Start the AI service:
    ```bash
    python app.py
    ```
    *   You should see: `Running on http://127.0.0.1:5000`

### 4. Frontend (Client) Setup
1.  Open a **new** terminal and navigate to the `client` directory:
    ```bash
    cd landing-zone-ai/client
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Open your browser and visit the URL shown (usually `http://localhost:5173`).

## Troubleshooting

- **Port Conflicts**: If port 3000, 5000, or 5173 are in use, update the `.env` files and `vite.config.js` accordingly.
- **MongoDB Connection**: Ensure the MongoDB service is running. You can use MongoDB Compass to verify the connection string `mongodb://localhost:27017`.
- **Python Imports**: If you get "Module not found" errors, ensure you have activated your virtual environment before running the app.
