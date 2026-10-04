# Task 4 — TODO: title from the assignment handout (e.g. web app / deployment)

**Author:** Ayesha Noor (i230736)

## Goal
TODO: one or two sentences from the Task 4 section of the handout.

## Overview
TODO: what the system does end to end (inputs, models used, outputs).

## Run locally
```bash
git clone https://github.com/Ayesha-Noor-04/genai-assignment1.git
cd genai-assignment1
pip install -r requirements.txt
# TODO: command that starts the web app, e.g. python app.py  /  streamlit run app.py
```
Then open `http://localhost:<port>`.

## Run with Docker
```bash
docker build -t genai-a1 .
docker run -p <port>:<port> genai-a1
```
Models (`*.onnx` and their `*.onnx.data` files) must be inside the image or mounted at `<path>`.

## Models used
Task 2 pipeline: classifier + salt-pepper / blur / occlusion specialists, served with ONNX Runtime.

## Links
- Deployed app: TODO
- Report: TODO
- Repository: https://github.com/Ayesha-Noor-04/genai-assignment1

## Notes
TODO