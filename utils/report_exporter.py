# utils/report_exporter.py
from fpdf import FPDF
import pandas as pd

def create_pdf_report(df_predictions, high_risk_info, charts_base64):
    """
    Generates a simple PDF report (R6).
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AAU Campus Security Risk Prediction Report", 0, 1, "C")
    
    # Date
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)

    # Section: High-Risk Predictions
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. High-Risk Summary", 0, 1)
    
    pdf.set_font("Arial", "", 12)
    for key, value in high_risk_info.items():
        pdf.cell(0, 7, f"- {key}: {value}", 0, 1)
    pdf.ln(5)

    # Section: Prediction Table
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2. Detailed Predictions", 0, 1)
    
    pdf.set_font("Arial", "B", 10)
    # Define column widths and headers
    col_width = [40, 40, 40, 40]
    headers = ['Location', 'Time (Hour)', 'Incident Type', 'Risk Level']
    for i, header in enumerate(headers):
        pdf.cell(col_width[i], 7, header, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    # Add prediction rows (limiting to top 15 for a concise report)
    for index, row in df_predictions.head(15).iterrows():
        pdf.cell(col_width[0], 6, str(row['Location']), 1, 0)
        pdf.cell(col_width[1], 6, str(row['Hour']), 1, 0, 'C')
        pdf.cell(col_width[2], 6, str(row['Predicted_Incident']), 1, 0)
        pdf.cell(col_width[3], 6, str(row['Risk_Level']), 1, 0, 'C')
        pdf.ln()
    pdf.ln(5)

    # Section: Charts (adding charts if available - may need adjustments for large images)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "3. Visual Analysis", 0, 1)
    
    # Note: Adding images in FPDF requires a saved file path, not base64.
    # For a real project, you would save the charts to 'reports/charts/' first.
    # For simplicity, we'll skip image embedding in this initial FPDF version.

    return pdf.output(dest='S').encode('latin-1') # Return bytes data for Streamlit download