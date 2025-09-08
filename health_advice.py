# health_advice.py
def generate_precautions(diagnosis_list):
    precautions = []

    for diag in diagnosis_list:
        diag_lower = diag.lower()
        
        if "fracture" in diag_lower:
            precautions.append("Avoid putting weight on the affected limb and rest it properly.")
            precautions.append("Use a cast or brace as prescribed by your doctor.")
            precautions.append("Attend follow-up appointments to monitor healing.")
        elif "osteoporosis" in diag_lower:
            precautions.append("Increase calcium and vitamin D intake.")
            precautions.append("Perform weight-bearing exercises as advised by your physician.")
            precautions.append("Avoid high-risk activities that may cause falls or fractures.")
        elif "arthritis" in diag_lower:
            precautions.append("Take anti-inflammatory medications as prescribed.")
            precautions.append("Perform gentle joint exercises to maintain mobility.")
            precautions.append("Avoid overstraining affected joints.")
        elif "no anomaly" in diag_lower:
            precautions.append("No major bone issues detected. Maintain a healthy lifestyle with exercise and balanced diet.")
        else:
            precautions.append(f"Condition '{diag}' detected. Please consult your doctor for specific advice.")

    return precautions
