

# 🛡️ AAU Campus Security Risk Analysis System

## Table of Contents

1.  [Project Overview](https://www.google.com/search?q=%23project-overview)
2.  [The History: Why We Built This System](https://www.google.com/search?q=%23the-history-why-we-built-this-system)
3.  [Goals and Functional Requirements](https://www.google.com/search?q=%23goals-and-functional-requirements)
4.  [System Architecture and Development Steps](https://www.google.com/search?q=%23system-architecture-and-development-steps)
5.  [Technology Stack](https://www.google.com/search?q=%23technology-stack)
6.  [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)

-----

## 1\. Project Overview

The **AAU Campus Security Risk Analysis System** is a predictive analytics dashboard designed to enhance safety at Ambrose Alli University (AAU), Ekpoma. Leveraging historical incident log data and Machine Learning (ML), the system forecasts future high-risk scenarios (locations, times, and incident types) to enable **proactive security deployment** rather than reactive response. The system is built for accessibility, ease of use, and low operational cost, making it ideal for university security operations.

-----

## 2\. The History: Why We Built This System

### 2.1 The Problem Statement (The "Why")

The motivation for designing this system stems from a critical challenge faced by many large campuses, including AAU Ekpoma: relying solely on **historical reporting** and **reactive security measures**.

  * **Reactive Approach:** Traditional security management involves analyzing incident reports *after* an event occurs. This approach is inherently limited; it manages consequence but does not prevent the event.
  * **Data Overload:** Security departments often collect large volumes of incident logs (location, time, type, response) but lack the tools to extract predictive insights from this data. The data remains siloed, primarily used for simple archival.
  * **Resource Misallocation:** Without predictive tools, security patrols and personnel deployment are often based on fixed schedules or anecdotal evidence, leading to inefficient use of limited resources. Incidents might occur in low-patrol areas or during specific time windows that were not adequately covered.
  * **The Need for Proactivity:** A critical need emerged to shift from *reporting* what happened to *predicting* what might happen, allowing security teams to place personnel, cameras, or alerts in the most vulnerable areas at the most dangerous times.

### 2.2 The Design Vision (The "What Makes Us Want to Design")

The system was conceived to be a **low-cost, data-driven security partner** fulfilling the following mandates:

1.  **Democratize Prediction:** Build an easy-to-use graphical interface (Streamlit) that allows non-technical security personnel to upload logs, retrain the model, and immediately view predictions and historical trends.
2.  **Focus on Local Context:** Use localized incident data (AAU/Ekpoma campus logs) to build a highly specific and relevant prediction model, rather than relying on generalized, non-contextual models.
3.  **Provide Actionable Intelligence:** The system must not just output a risk score, but a clear, summarized forecast (e.g., "High Risk at **Hostel D Gate** between **2:00 AM - 4:00 AM** for **Theft**") that directly informs patrol schedules and resource deployment.
4.  **Integrate External Context:** Include supplementary information, such as real-time news related to security in the surrounding Ekpoma area, to add necessary contextual awareness to the predicted risks.

-----

## 3\. Goals and Functional Requirements

The system design adheres to the following core goals (Requirements) and capabilities (Features):

| ID | Requirement Category | Description |
| :--- | :--- | :--- |
| **R1** | **Data Management** | Must allow users to easily **upload and load** incident log data in CSV format. |
| **R2** | **Trend Analysis** | Must visually represent **historical incident trends** (e.g., bar charts, frequency graphs) for various filters (location, type, date). |
| **R3** | **Risk Prediction** | Must utilize a Machine Learning model to forecast **high-risk scenarios** (combination of location, time, and incident type). |
| **R4** | **Summary Metrics** | Must provide **top-level summary metrics** of the prediction (e.g., Top High-Risk Location, Most Probable Incident Type). |
| **R5** | **Contextual Awareness**| Must fetch and display **supplementary news** from external sources related to campus/Ekpoma security. |
| **R6** | **Reporting** | Must allow the user to **export a PDF report** detailing the latest risk forecast. |
| **FR7** | **Heatmap Visualization**| The core visualization must include an **Incident Frequency Heatmap** showing `Location` vs. `Hour`. |
| **FR8** | **Filtering** | All historical trends and visualizations must be **filterable** by location, incident type, and year. |
| **FR9** | **Model Retraining** | Must include functionality for security staff to **retrain the ML model** with new, updated log data via the user interface. |

-----

## 4\. System Architecture and Development Steps

The system uses a modular architecture, promoting maintainability and clear separation of concerns.

### 4.1 Architecture Diagram

### 4.2 Development Steps

The development process followed a clear pipeline:

1.  **Data Acquisition & Preparation (`utils/preprocessing.py`)**:

      * **Goal:** Clean raw CSV logs, handle missing values, and convert date/time strings into usable datetime objects.
      * **Step:** Implement the `load_data()` function to perform data validation and type conversion.
      * **Step:** Implement the `feature_engineering()` function to extract temporal features (Hour, DayOfWeek, Month) and encode categorical features for the ML model.

2.  **Model Development (`model/train.py`, `model/predictor.py`)**:

      * **Goal:** Build, train, and save the predictive model (Random Forest Classifier).
      * **Step:** Implement `train_model()` to load processed data, select features, fit the Random Forest algorithm, and save the model (`.pkl`) and the feature encoder (`.pkl`).
      * **Step:** Implement `predict_risks()` to load the saved model and encoder, generate prediction samples for the future, and output high-risk locations/times.

3.  **Visualization & Reporting (`utils/visualization.py`, `utils/report_exporter.py`)**:

      * **Goal:** Create visual representations of historical data and export reports.
      * **Step:** Implement `create_plotly_heatmap()` and `create_bar_chart()` to display key insights.
      * **Step:** Implement `create_pdf_report()` to compile prediction and summary metrics into a professional PDF document (R6).

4.  **Frontend & Integration (`app/app.py`)**:

      * **Goal:** Build the interactive dashboard and integrate all backend components.
      * **Step:** Set up the Streamlit interface with tabs for Trends, Prediction, and News.
      * **Step:** Implement the data upload (R1) and model retraining workflow (FR9).
      * **Step:** Integrate the `sys.path` fix for robust module imports in Streamlit Cloud environments.

-----

## 5\. Technology Stack

The system relies on a focused stack of accessible, open-source Python libraries.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Dashboard** | **Streamlit** | Primary framework for the web application interface. |
| **Machine Learning**| **Scikit-learn (Random Forest)** | Algorithm used for classification/prediction of incident type and risk level. |
| **Data Processing** | **Pandas / NumPy** | Core libraries for data cleaning, manipulation, and numerical operations. |
| **Visualization** | **Plotly Express** | Used to create interactive, informative charts (Heatmaps, Bar Charts) (FR7). |
| **Reporting** | **fpdf** | Library used to generate and export structured PDF risk reports (R6). |
| **News Fetching** | **feedparser** | Used to pull and cache external news articles for contextual awareness (R5). |

-----

## 6\. Setup and Installation

### A. Prerequisites

  * Python 3.9+
  * Git

### B. Local Installation Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/YourUsername/campus-incident-detection-system.git
    cd campus-incident-detection-system
    ```

2.  **Create and Activate Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies:**
    Make sure your `requirements.txt` file includes **`streamlit`**, **`pandas`**, **`scikit-learn`**, **`plotly`**, and **`feedparser`**.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**

    ```bash
    streamlit run app/app.py
    ```

The application will open in your browser, ready for use.