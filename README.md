# Network Alarm Prediction System

[![Theme](https://img.shields.io/badge/Theme-AI/ML%20%7C%20Open%20Innovation-blueviolet)](https://shields.io/)

An intelligent system to predict the next network alarm based on historical and current alarm data from large-scale optical communication networks. This project shifts from a reactive to a proactive approach in handling network faults, aiming to reduce hardware failures, improve SLA compliance, and maximize network uptime.

## 🚀 Overview

In large-scale optical communication networks, alarms are critical indicators of faults or anomalies. Handling these reactively leads to service disruptions and increased maintenance costs. This project introduces a predictive system that analyzes historical alarm data to forecast future alarms, identify patterns, and suggest preventive actions. By anticipating potential failures, such as a site failing due to heat-related issues, we can transform network maintenance from a reactive to a proactive, game-changing strategy.

*Image: Welcome page of the Network Alarm Prediction System showing key performance metrics like Prediction Accuracy, Response Time, and Downtime Prevention.*

## ✨ Key Features

Our Streamlit application provides an end-to-end workflow for network alarm prediction:

* **📤 Data Upload & Exploration**: Upload your own network alarm data in CSV format or use our sample dataset to get started. Preview the data, view summary statistics, and check data quality.
    *Image: Data Upload page with a drag-and-drop area and a preview of the alarm data table.*
 <img width="1905" height="1181" alt="image" src="https://github.com/user-attachments/assets/d8268b38-db5b-41c7-9058-b44e1b86fbe8" />



* **📊 Exploratory Data Analysis (EDA)**: Get a high-level overview of your data with key metrics like total alarms, critical alarms, and the most affected sites. Analyze alarm distributions, time trends, and correlations.
    *Image: EDA dashboard showing cards for key metrics and a bar chart of alarm distribution by type.*
  <img width="1910" height="1195" alt="image" src="https://github.com/user-attachments/assets/daf1c203-d056-48e5-bfef-5fcc7fc3ac7b" />


* **🤖 ML Model Training**: Train a suite of machine learning models through a chained pipeline to predict various aspects of the next alarm, including its type, probable cause, duration, severity, and timestamp.
    *Image: Model Training page showing a summary of trained models, including classifiers and regressors.*
  <img width="1913" height="1191" alt="image" src="https://github.com/user-attachments/assets/33e44b52-f8ff-47da-8f83-d877e8a23b66" />

  

* **🔮 Alarm Predictions**: Configure and run multi-step predictions for specific network sites. Get detailed forecasts for the next alarm's attributes along with a confidence score.
    *Image: Predictions page displaying the forecasted Alarm Type, Cause, and Severity for a selected site.*
 <img width="1908" height="1074" alt="image" src="https://github.com/user-attachments/assets/5f2c46f2-6496-4578-9886-a7b640d3b2f2" />



* **📈 Advanced Visualizations**: Dive deep into alarm patterns with interactive visualizations like network alarm hotspot maps to identify high-intensity sites.
    *Image: Visualizations page showing a heatmap of alarm intensity across different network sites.*
  <img width="1906" height="1177" alt="image" src="https://github.com/user-attachments/assets/c39c680b-c6d6-458d-9304-86785efc6581" />



* **💡 Insights & Root Cause Analysis**: Explore frequent alarms, site-specific behaviors, and potential root causes to understand the underlying issues in the network.
    *Image: Insights page showing a site-specific analysis with total alarms and alarm types.*
  <img width="1919" height="1103" alt="image" src="https://github.com/user-attachments/assets/5c287856-8507-4e7c-80fa-9d536537b78e" />



* **🏢 System Architecture**: An interactive diagram showcasing the end-to-end data flow from ingestion and preprocessing to model training, prediction, and alerting.
    *Image: A diagram illustrating the system architecture, from data ingestion to the prediction engine and dashboard.*
  <img width="1916" height="1128" alt="image" src="https://github.com/user-attachments/assets/b67b7a5f-0847-4b21-863d-74112c5ae2bd" />

  

* **🔥 Live Demo Simulation**: Experience the system's capabilities with a real-time crisis simulation, such as a "Summer Heatwave," to monitor and manage alarms proactively.
    *Image: A simulation page displaying a real-time scatter plot of alarm events during a heatwave scenario.*
  <img width="1918" height="1194" alt="image" src="https://github.com/user-attachments/assets/e35b6f9f-f8b3-433a-ba6e-a7d405661a7f" />


## 🛠️ Tech Stack

This project leverages a modern tech stack to deliver a robust and scalable solution:

* **Backend**: Streamlit
* **Frontend**: Streamlit,React
* **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy

## ⚙️ How It Works

The application follows a simple and intuitive workflow:

1.  **Upload Data**: Start by uploading your historical network alarm data.
2.  **Explore**: Use the EDA tools to understand patterns and correlations in your data.
3.  **Train Models**: Train the prediction models on your dataset with a single click.
4.  **Visualize**: Analyze trends and identify hotspots using advanced visualizations.
5.  **Predict & Act**: Generate predictions for future alarms and receive proactive alerts to prevent issues.

## 📦 Installation and Setup

To get this project up and running on your local machine, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/network-alarm-prediction.git](https://github.com/your-username/network-alarm-prediction.git)
    cd network-alarm-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

