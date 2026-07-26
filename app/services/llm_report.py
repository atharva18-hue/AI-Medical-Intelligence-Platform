"""
Medical report generation using LLM.
Uses OpenAI if API key is set, otherwise falls back to template.
This way project works even without paid API during demo.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def generate_report_template(prediction: dict, patient_notes: str = "") -> str:
    """Rule-based report - backup when no OpenAI key"""
    disease = prediction["predicted_class"]
    conf = prediction["confidence"]
    probs = prediction["probabilities"]

    report = f"""MEDICAL IMAGING REPORT
{'='*40}

FINDINGS:
Chest X-ray analysis was performed using our deep learning model.
Primary finding: {disease}
Model confidence: {conf}%

Detailed probabilities:
  - Normal: {probs.get('Normal', 0)}%
  - Pneumonia: {probs.get('Pneumonia', 0)}%

"""

    if disease == "Pneumonia":
        report += """
IMPRESSION:
Findings are suggestive of PNEUMONIA. Opacities may be present in lung fields.
Recommend clinical correlation and follow-up imaging if symptoms persist.

RECOMMENDATION:
- Consult physician for further evaluation
- Consider antibiotic therapy if clinically indicated
- Repeat chest X-ray in 2-4 weeks if needed
"""
    else:
        report += """
IMPRESSION:
No significant abnormality detected. Lung fields appear within normal limits
based on automated analysis.

RECOMMENDATION:
- Routine follow-up as per clinical protocol
- Correlation with patient symptoms advised
"""

    if patient_notes:
        report += f"\nCLINICAL NOTES (provided):\n{patient_notes}\n"

    report += """
DISCLAIMER: This is an AI-assisted preliminary analysis tool.
Final diagnosis must be made by a qualified medical professional.
"""
    return report


async def generate_medical_report(prediction: dict, patient_notes: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # no key = use template (works for college demo)
        return generate_report_template(prediction, patient_notes)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""You are a radiology assistant. Write a brief medical report based on this AI chest X-ray analysis:

Prediction: {prediction['predicted_class']}
Confidence: {prediction['confidence']}%
Probabilities: {prediction['probabilities']}
Patient notes: {patient_notes or 'None provided'}

Write a professional but concise report with FINDINGS, IMPRESSION, and RECOMMENDATION sections.
Add disclaimer that this is AI-assisted and needs doctor verification."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        # if API fails just use template - dont break the app
        print(f"LLM error: {e}, using template fallback")
        return generate_report_template(prediction, patient_notes)
