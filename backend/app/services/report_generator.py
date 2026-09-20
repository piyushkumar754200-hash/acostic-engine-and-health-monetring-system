import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from app.config import REPORTS_DIR, APP_NAME

class ReportGenerator:
    @staticmethod
    def generate_pdf_report(analysis_data: dict, filepath: str = None) -> str:
        analysis_id = analysis_data.get("analysis_id", "ANL-001")
        if filepath is None:
            filepath = os.path.join(REPORTS_DIR, f"EngineSense_Report_{analysis_id}.pdf")

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom ReportLab Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=6
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#64748B'),
            spaceAfter=15
        )

        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            textColor=colors.HexColor('#1E293B'),
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#334155'),
            leading=13
        )

        disclaimer_style = ParagraphStyle(
            'ReportDisclaimer',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=colors.HexColor('#94A3B8'),
            leading=11
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("ENGINE-SENSE AI — DIAGNOSTIC REPORT", title_style))
        story.append(Paragraph(f"Analysis ID: {analysis_id} | Generated: {analysis_data.get('timestamp', '')}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceAfter=15))

        # 2. Overall Diagnosis Box
        pred_condition = analysis_data.get("predicted_condition", "UNKNOWN")
        confidence = analysis_data.get("overall_confidence", 0.0) * 100.0
        severity = analysis_data.get("severity", "Medium")

        is_normal = "normal" in pred_condition.lower()
        banner_color = colors.HexColor('#10B981') if is_normal else colors.HexColor('#EF4444')
        
        diag_html = f"<b>OVERALL DIAGNOSIS:</b> <font color='{banner_color.hexval()}'><b>{pred_condition.upper()}</b></font><br/>" \
                    f"<b>Confidence Score:</b> {confidence:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; <b>Severity Level:</b> {severity}"
        
        diag_table = Table([[Paragraph(diag_html, body_style)]], colWidths=[540])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BORDER', (0, 0), (-1, -1), 1, banner_color),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 15))

        # 3. Vehicle & Audio Details Table
        story.append(Paragraph("Vehicle & Recording Metadata", heading_style))
        vehicle = analysis_data.get("vehicle_info", {})
        
        meta_data = [
            [Paragraph("<b>Vehicle Brand:</b>", body_style), Paragraph(str(vehicle.get("brand", "N/A")), body_style),
             Paragraph("<b>Audio File:</b>", body_style), Paragraph(str(analysis_data.get("filename", "N/A")), body_style)],
            [Paragraph("<b>Model / Engine:</b>", body_style), Paragraph(f"{vehicle.get('model', '')} ({vehicle.get('engine_type', '')})", body_style),
             Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{analysis_data.get('duration_seconds', 5.0):.2f} sec", body_style)],
            [Paragraph("<b>Fuel / Mileage:</b>", body_style), Paragraph(f"{vehicle.get('fuel_type', '')} / {vehicle.get('mileage', 0)} km", body_style),
             Paragraph("<b>Processing Time:</b>", body_style), Paragraph(f"{analysis_data.get('processing_time_seconds', 0.0):.3f} sec", body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))

        # 4. Multi-Model Predictions Table
        story.append(Paragraph("Multi-Model Predictions Breakdown", heading_style))
        model_preds = analysis_data.get("model_predictions", {})
        
        model_table_data = [
            [Paragraph("<b>Model</b>", body_style), Paragraph("<b>Type</b>", body_style), Paragraph("<b>Predicted Class</b>", body_style), Paragraph("<b>Confidence</b>", body_style)]
        ]

        for m_id, m_res in model_preds.items():
            name = m_res.get("model_name", m_id)
            m_type = m_res.get("type", "ML/DL")
            p_class = m_res.get("prediction", "N/A")
            conf = m_res.get("confidence", 0.0) * 100.0
            model_table_data.append([
                Paragraph(name, body_style),
                Paragraph(m_type, body_style),
                Paragraph(p_class, body_style),
                Paragraph(f"{conf:.1f}%", body_style)
            ])

        model_table = Table(model_table_data, colWidths=[160, 120, 150, 110])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(model_table)
        story.append(Spacer(1, 15))

        # 5. Acoustic Waveform Figure
        story.append(Paragraph("Acoustic Waveform Analysis", heading_style))
        viz_data = analysis_data.get("visualizations", {})
        waveform = viz_data.get("waveform", [])
        
        if waveform:
            fig, ax = plt.subplots(figsize=(7, 1.8), dpi=100)
            ax.plot(waveform, color='#0284C7', linewidth=1)
            ax.set_facecolor('#F8FAFC')
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_title("Engine Sound Time-Domain Waveform", fontsize=9, color='#334155')
            ax.set_xlabel("Time Frames", fontsize=8, color='#64748B')
            ax.set_ylabel("Amplitude", fontsize=8, color='#64748B')
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            plt.close(fig)
            img_buf.seek(0)
            
            story.append(Image(img_buf, width=540, height=135))
            story.append(Spacer(1, 15))

        # 6. Recommendation & Disclaimer
        story.append(Paragraph("Mechanical Recommendation", heading_style))
        rec_text = analysis_data.get("recommendation", "Inspect engine components regularly.")
        story.append(Paragraph(rec_text, body_style))
        story.append(Spacer(1, 15))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=10))
        disclaimer = "DISCLAIMER: This system provides an AI-assisted acoustic analysis and should not be treated as a substitute for professional mechanical inspection."
        story.append(Paragraph(disclaimer, disclaimer_style))

        doc.build(story)
        return filepath
