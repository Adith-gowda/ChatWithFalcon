# ChatWithFalcon - Falcon 7B Instruct Chatbot

Falcon 7B Instruct Chatbot is an advanced conversational AI designed to provide up-to-date answers using current data, overcoming limitations of older models like ChatGPT 3.5 ( Overcomed by ChatGPT-4o ). It integrates the Falcon 7B Instruct model and RAG Teachnique for knowledge retrieval and answering user queries based on real-time information.

## Table of Contents

- [Demo]
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Frontend Setup](#frontend-setup)
  - [Backend Setup with FAST API and NGROK](#backend-setup-with-fast-api-and-ngrok)
- [Ehancements]

## Demo

Watch the demo video below:

[Watch the video](https://drive.google.com/file/d/1WS44zNSthMuTeBhWsE-WUNv3nru-4IwA/view?usp=sharing)

## Overview

Falcon 7B Instruct Chatbot leverages advanced AI models to deliver accurate and current information in response to user queries. It utilizes the Falcon 7B Instruct model for contextual understanding and the RAG model for efficient data retrieval. This README provides a guide to setting up and using the chatbot locally.

## Features

- **Real-time Data Retrieval:** Fetches current information from Wikipedia based on user query keywords.
- **AI Models:** Integrates Falcon 7B Instruct model for understanding context and RAG model for retrieving relevant data.
- **Responsive Frontend:** Built using React JS for an intuitive user interface.
- **Scalable Backend:** Powered by FAST API and NGROK for seamless communication between frontend and backend.
- **GPU Acceleration:** Utilizes Google Colab for GPU-accelerated processing of backend operations.

## Technologies Used

- **Frontend:** React JS
- **Backend:** FastAPI and NGROK
- **Frameworks:** LangChain, RAG (Retrieval-Augmented Generation) technique and FastAPI
- **AI Models:** Falcon 7B Instruct model and multilingual-e5-large-instruct model.
- **Database:** FAISS (Facebook AI Similarity Search)
- **Deployment:** Google Colab for backend GPU processing

## Installation

### Frontend Setup

To run the frontend of Falcon 7B Instruct Chatbot:

1. Clone the repository:
   ```bash
   git clone https://github.com/Adith-gowda/ChatWithFalcon.git
   cd ChatWithFalcon
2. Install Node Modules
   ```bash
   npm install
3. Start the development server
   ```bash
   npm start

### Backend Setup with FAST API and NGROK

To run the backend, 

1. Download the IPYNB file ( python notebook ) and run it in GOOGLE COLAB by connecting to T4 runtime.
2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken and replace that in respective cell in python notebook.

   NOTE: NGROK provides a secure way to expose your local FAST API server to the internet temporarily. This is useful for testing and development purposes.

3. Update the Axios calls in the frontend to point to your NGROK URL for backend API access. ( In falcon.js file you need to update the NGROK URL )

## Enhancements

1. Adding Conversational RAG ( Conversational Chain with RAG )
2. Using non quantized Falcon 7B/13B/180B.
3. Enhancing RAG features like using MultiQueryRetriever RAG chain
