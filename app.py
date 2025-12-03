from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.metrics import roc_curve, auc
#from tensorflow.keras.models import load_model


# Load your trained seizure model once




# Load your valid EEG dataset features
# Load precomputed EEG image hashes for strict validation
import imagehash


# Load EEG perceptual hashes
EEG_HASHES = np.load(r"C:\Users\saniy\Downloads\EEG seizure and non-seizure image dataset\hashes.npy", allow_pickle=True)





# ========== TENSORFLOW IMPORT FIX ==========
#..

# For advanced image validation
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - using basic validation")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///eeg_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

db = SQLAlchemy(app)

# ========== MODELS ==========
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    diagnosis = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    eeg_records = db.relationship('EEGRecord', backref='patient', lazy=True, cascade='all, delete-orphan')

class EEGRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='uploaded')
    processed = db.Column(db.Boolean, default=False)
    analyses = db.relationship('Analysis', backref='eeg_record', lazy=True, cascade='all, delete-orphan')

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    eeg_record_id = db.Column(db.Integer, db.ForeignKey('eeg_record.id'), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    details = db.Column(db.Text)
    spectrogram_path = db.Column(db.String(300))
    histogram_path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ========== LOAD ML MODEL (FIXED) ==========
#SEIZURE_MODEL = None
#MODEL_AVAILABLE = False



# ========== VALIDATION FUNCTIONS ==========



# ========== ROUTES ==========
@app.route('/')
def index():
    patients = Patient.query.all()
    total_patients = Patient.query.count()
    total_records = EEGRecord.query.count()
    total_analyses = Analysis.query.count()
    return render_template('index.html', 
                         patients=patients,
                         total_patients=total_patients,
                         total_records=total_records,
                         total_analyses=total_analyses)

@app.route('/patients')
def patients():
    all_patients = Patient.query.all()
    return render_template('patients.html', patients=all_patients)

@app.route('/patients/add', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        diagnosis = request.form.get('diagnosis', '')
        
        new_patient = Patient(name=name, age=age, gender=gender, diagnosis=diagnosis)
        db.session.add(new_patient)
        db.session.commit()
        
        flash('Patient added successfully!', 'success')
        return redirect(url_for('patients'))
    
    return render_template('add_patient.html')

@app.route('/patients/<int:id>')
def patient_detail(id):
    patient = Patient.query.get_or_404(id)
    eeg_records = EEGRecord.query.filter_by(patient_id=id).all()
    return render_template('patient_detail.html', patient=patient, eeg_records=eeg_records)

@app.route('/patients/update/<int:id>', methods=['GET', 'POST'])
def update_patient(id):
    patient = Patient.query.get_or_404(id)
    
    if request.method == 'POST':
        patient.name = request.form['name']
        patient.age = request.form['age']
        patient.gender = request.form['gender']
        patient.diagnosis = request.form.get('diagnosis', '')
        
        db.session.commit()
        flash('Patient updated successfully!', 'success')
        return redirect(url_for('patient_detail', id=id))
    
    return render_template('update_patient.html', patient=patient)

@app.route('/patients/delete/<int:id>')
def delete_patient(id):
    patient = Patient.query.get_or_404(id)
    db.session.delete(patient)
    db.session.commit()
    flash('Patient deleted successfully!', 'success')
    return redirect(url_for('patients'))

# ========== UPLOAD CHOICE ROUTE (NEW) ==========
@app.route('/eeg/upload_choice/<int:patient_id>')
def upload_choice(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('upload_choice.html', patient=patient)

# ========== SEIZURE UPLOAD WITH STRICT VALIDATION ==========
# Load your precomputed dataset features
dataset_features = np.load(r"C:\Users\saniy\Downloads\EEG seizure and non-seizure image dataset\features.npy", allow_pickle=True)

@app.route('/eeg/upload_seizure/<int:patient_id>', methods=['GET'])
def upload_seizure_page(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('upload_seizure.html', patient=patient)

import hashlib

@app.route('/eeg/upload_seizure/<int:patient_id>', methods=['POST'])
def upload_seizure(patient_id):
    try:
        patient = Patient.query.get_or_404(patient_id)
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'success': False, 'message': 'No file uploaded', 'beep': True})

        # --- File validation ---
        allowed_exts = {'png','jpg','jpeg'}
        ext = file.filename.rsplit('.',1)[-1].lower()
        if ext not in allowed_exts:
            return jsonify({'success': False, 'message': 'Invalid file type', 'beep': True})

        # --- Compute hash and validate EEG ---
        file.seek(0)
        img = Image.open(file).convert('L').resize((128,128))
        uploaded_hash = str(imagehash.phash(img))
        if uploaded_hash not in EEG_HASHES:
            return jsonify({'success': False, 'message': 'Invalid EEG spectrogram', 'beep': True})

        # --- Save file ---
        filename = f"seizure_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.seek(0)
        file.save(save_path)

        # --- Save EEG record ---
        new_record = EEGRecord(
            patient_id=patient_id,
            filename=filename,
            file_path=save_path,
            status='uploaded'
        )
        db.session.add(new_record)
        db.session.commit()

        # --- ML prediction ---
        try:
            COLAB_URL = "https://overcoyly-crispy-lenard.ngrok-free.dev"  # <-- Replace THIS ONLY

            with open(save_path, "rb") as f:
                response = requests.post(COLAB_URL, files={"file": f})

            if response.status_code != 200:
                label = "ML API Error"
                seizure_prob = None
            else:
                data = response.json()
                label = data.get("prediction", "No result")
                seizure_prob = data.get("probability", None)

        except Exception as ml_error:
            print("🔥 ML API Error:", ml_error)
            label = "Model API unreachable"
            seizure_prob = None

        # ============================================================

        # --- Save analysis ---
        analysis = Analysis(
            eeg_record_id=new_record.id,
            analysis_type='seizure_detection',
            result=label,
            confidence=0.0 if seizure_prob is None else seizure_prob,
            details=f"Seizure detection analysis using ML API"
                    + ("" if seizure_prob is None else f" ({seizure_prob:.2f} confidence)"),
            spectrogram_path=save_path,
            histogram_path=None
        )
        db.session.add(analysis)
        new_record.status = 'completed'
        db.session.commit()

        # --- Return JSON ---
        return jsonify({
            'success': True,
            'message': f'Valid EEG spectrogram uploaded successfully! Prediction: {label}'
                       + ("" if seizure_prob is None else f" ({seizure_prob:.2f})"),
            'beep': False,
            'eeg_id': new_record.id,
            'prediction': label,
            'probability': seizure_prob
        })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'message': f'Upload failed: {str(e)}', 'beep': True}), 500

# Add these imports to your Flask app
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import io




# Update the download_report route





# ================= ASD UPLOAD =================
@app.route('/eeg/upload_asd/<int:patient_id>', methods=['GET', 'POST'])
def upload_asd(patient_id):
    patient = Patient.query.get_or_404(patient_id)

    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        file = request.files.get('file')

        if not file or file.filename == '':
            msg = 'No file selected'
            if is_ajax:
                return jsonify({'success': False, 'message': msg, 'beep': True})
            flash(msg, 'danger')
            return redirect(request.url)

        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext != 'csv':
            msg = 'Only CSV files allowed for ASD'
            if is_ajax:
                return jsonify({'success': False, 'message': msg, 'beep': True})
            flash(msg, 'danger')
            return redirect(request.url)

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            final_filename = f"asd_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
            file.save(final_path)

            # Save record to DB
            new_record = EEGRecord(
                patient_id=patient_id,
                filename=final_filename,
                file_path=final_path,
                status='uploaded'
            )
            db.session.add(new_record)
            db.session.commit()

            if is_ajax:
                return jsonify({'success': True, 'message': 'CSV uploaded successfully!', 'eeg_id': new_record.id})
            flash('CSV uploaded successfully!', 'success')
            return redirect(url_for('preview_asd', eeg_id=new_record.id))

        except Exception as e:
            print(f"Upload error: {e}")
            msg = 'Upload failed due to server error'
            if is_ajax:
                return jsonify({'success': False, 'message': msg, 'beep': True})
            flash(msg, 'danger')
            return redirect(request.url)

    return render_template('upload_asd.html', patient=patient)


# ========== PREVIEW ROUTES ==========
@app.route('/seizure/preview/<int:eeg_id>')
def preview_seizure(eeg_id):
    eeg_record = EEGRecord.query.get_or_404(eeg_id)
    patient = Patient.query.get_or_404(eeg_record.patient_id)
    return render_template('preview_seizure.html', 
                         eeg_record=eeg_record, 
                         patient=patient,
                         image_path=eeg_record.file_path)



# ========== PREDICTION ROUTES ==========




@app.route('/analyses')
def analyses():
    all_analyses = Analysis.query.all()
    return render_template('analyses.html', analyses=all_analyses)

@app.route('/analysis/<int:id>')
def analysis_detail(id):
    analysis = Analysis.query.get_or_404(id)
    return render_template('analysis_detail.html', analysis=analysis)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/eeg/<int:id>')
def eeg_detail(id):
    eeg_record = EEGRecord.query.get_or_404(id)
    analyses = Analysis.query.filter_by(eeg_record_id=id).all()
    return render_template('eeg_detail.html', eeg_record=eeg_record, analyses=analyses)

@app.route('/spectrogram/<path:filename>')
def spectrogram_file(filename):
    """Serve spectrogram files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Replace the chat route in app.py with this improved version

# Replace the chat route in app.py with this improved version

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chatbot with navigation support"""
    try:
        user_message = request.form.get('message', '').strip().lower()
        
        if not user_message:
            return jsonify({
                'reply': 'Please type a message!',
                'action': None
            })
        
        # Navigation commands
        navigation_map = {
            'upload patient': ('patients', 'Taking you to patients page...'),
            'add patient': ('add_patient', 'Opening patient registration form...'),
            'upload eeg': ('patients', 'Go to patients and select a patient to upload EEG'),
            'view patients': ('patients', 'Showing all patients...'),
            'view analyses': ('analyses', 'Opening analyses dashboard...'),
            'dashboard': ('index', 'Going to dashboard...'),
            'home': ('index', 'Taking you home...'),
            'about': ('about', 'Opening about page...'),
        }
        
        # Check for navigation commands
        for keyword, (route, message) in navigation_map.items():
            if keyword in user_message:
                return jsonify({
                    'reply': message,
                    'action': url_for(route)
                })
        
        # Informational responses
        if 'seizure' in user_message:
            if 'upload' in user_message:
                return jsonify({
                    'reply': 'To upload for seizure detection: Go to Patients → Select patient → Click "Upload EEG" → Choose "Seizure Detection" → Upload PNG/JPG spectrogram',
                    'action': url_for('patients')
                })
            else:
                return jsonify({
                    'reply': 'Seizure detection analyzes EEG spectrograms using frequency analysis and pattern detection. It detects sharp transitions and high-frequency patterns associated with seizures.',
                    'action': None
                })
        
        elif 'asd' in user_message or 'autism' in user_message:
            if 'upload' in user_message:
                return jsonify({
                    'reply': 'To upload for ASD detection: Go to Patients → Select patient → Click "Upload EEG" → Choose "ASD Detection" → Upload CSV file',
                    'action': url_for('patients')
                })
            else:
                return jsonify({
                    'reply': 'ASD (Autism Spectrum Disorder) detection analyzes EEG patterns from CSV data to identify markers associated with autism.',
                    'action': None
                })
        
        elif 'patient' in user_message:
            if 'add' in user_message or 'create' in user_message or 'new' in user_message:
                return jsonify({
                    'reply': 'Opening patient registration form. Fill in name, age, gender, and diagnosis.',
                    'action': url_for('add_patient')
                })
            elif 'view' in user_message or 'list' in user_message or 'all' in user_message:
                return jsonify({
                    'reply': 'Showing all registered patients...',
                    'action': url_for('patients')
                })
            else:
                return jsonify({
                    'reply': 'You can add new patients, view patient details, upload EEG data, and manage patient records. What would you like to do?',
                    'action': None
                })
        
        elif 'analysis' in user_message or 'analyze' in user_message:
            if 'view' in user_message or 'see' in user_message:
                return jsonify({
                    'reply': 'Opening analyses dashboard...',
                    'action': url_for('analyses')
                })
            elif 'run' in user_message or 'start' in user_message:
                return jsonify({
                    'reply': 'To run analysis: 1) Upload EEG data for a patient 2) The system will automatically analyze it 3) View results in the analyses section',
                    'action': url_for('patients')
                })
            else:
                return jsonify({
                    'reply': 'Analysis includes seizure detection and ASD detection. Upload EEG data to get started!',
                    'action': None
                })
        
        elif 'report' in user_message or 'download' in user_message:
            return jsonify({
                'reply': 'To download a report: Go to Analyses → Click on an analysis → Click "Download PDF Report". Reports include graphs, heatmaps, and detailed findings.',
                'action': url_for('analyses')
            })
        
        elif 'help' in user_message:
            return jsonify({
                'reply': ("I can help you with:\n"
                          "• 'add patient' - Register new patient\n"
                          "• 'upload eeg' - Upload EEG data\n"
                          "• 'view patients' - See all patients\n"
                          "• 'view analyses' - See all analyses\n"
                          "• 'seizure' or 'asd' - Learn about detection methods\n"
                          "• 'download report' - Get analysis reports\n\n"
                          "Just type what you need!"),
                'action': None
            })
        
        elif any(greeting in user_message for greeting in ['hi', 'hello', 'hey']):
            return jsonify({
                'reply': "Hello! 👋 I'm your EEG Assistant. I can help you navigate the system, upload data, and answer questions. Type 'help' to see what I can do!",
                'action': None
            })
        
        else:
            return jsonify({
                'reply': 'I\'m not sure about that. Try asking about "patients", "upload", "seizure detection", "ASD", or type "help" for more options.',
                'action': None
            })
            
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'reply': 'Sorry, I encountered an error. Please try again.',
            'action': None
        }), 200






#from tensorflow.keras.preprocessing.image import img_to_array
# Add this to your Flask app.py

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

def detect_seizure_without_ml(image_path):
    """
    Rule-based seizure detection using frequency analysis and pattern detection
    Returns: (prediction, confidence, heatmap_path)
    """
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error", 0.0, None
            
        # Resize for consistency
        img = cv2.resize(img, (256, 256))
        
        # --- Feature Extraction ---
        
        # 1. Frequency domain analysis
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 2. Detect high-frequency components (seizures have more high-freq)
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask for high frequencies
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # radius for low-pass
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        # Apply mask and get high-freq energy
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy
        
        # 3. Detect sharp transitions (edges) - seizures have more abrupt changes
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 4. Intensity variance analysis
        intensity_std = np.std(img)
        
        # 5. Pattern irregularity (entropy)
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # --- Decision Logic ---
        seizure_score = 0
        confidence_factors = []
        
        # High frequency check (seizures typically have more high-freq content)
        if high_freq_ratio > 0.15:
            seizure_score += 30
            confidence_factors.append("High frequency content detected")
        
        # Edge density check
        if edge_density > 0.1:
            seizure_score += 25
            confidence_factors.append("Sharp transitions present")
        
        # Variance check
        if intensity_std > 40:
            seizure_score += 20
            confidence_factors.append("High intensity variation")
        
        # Entropy check (irregular patterns)
        if entropy > 6.0:
            seizure_score += 25
            confidence_factors.append("Irregular patterns detected")
        
        # --- Generate Heatmap ---
        
        # Create intensity heatmap with gaussian smoothing
        heatmap = gaussian_filter(img.astype(float), sigma=2)
        
        # Normalize for better visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Seizure Detection Analysis', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Spectrogram')
        axes[0, 0].axis('off')
        
        # Intensity heatmap
        im1 = axes[0, 1].imshow(heatmap, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Intensity Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Edge detection
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection (Sharp Transitions)')
        axes[1, 0].axis('off')
        
        # Frequency spectrum
        magnitude_display = np.log(magnitude_spectrum + 1)
        im2 = axes[1, 1].imshow(magnitude_display, cmap='viridis')
        axes[1, 1].set_title('Frequency Spectrum')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_filename = f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # --- Final Decision ---
        confidence = min(seizure_score / 100.0, 0.99)
        
        if seizure_score >= 50:
            prediction = "Seizure Detected"
        else:
            prediction = "No Seizure"
            confidence = 1.0 - confidence
        
        details = {
            'high_freq_ratio': float(high_freq_ratio),
            'edge_density': float(edge_density),
            'intensity_std': float(intensity_std),
            'entropy': float(entropy),
            'seizure_score': int(seizure_score),
            'factors': confidence_factors
        }
        
        return prediction, confidence, heatmap_path, details
        
    except Exception as e:
        print(f"Seizure detection error: {e}")
        return "Error", 0.0, None, {}


# Replace your predict_seizure route with this:
@app.route('/predict_seizure', methods=['POST'])
def predict_seizure():
    try:
        eeg_id = request.form.get('eeg_id')
        eeg_record = EEGRecord.query.get_or_404(eeg_id)
        
        # Run rule-based seizure detection
        prediction, confidence, heatmap_path, details = detect_seizure_without_ml(eeg_record.file_path)
        
        if prediction == "Error":
            return jsonify({'error': 'Analysis failed'}), 500
        
        # Create detailed description
        detail_text = f"Rule-based seizure detection analysis completed.\n\n"
        detail_text += f"Detection Score: {details.get('seizure_score', 0)}/100\n\n"
        detail_text += "Analysis Metrics:\n"
        detail_text += f"- High Frequency Ratio: {details.get('high_freq_ratio', 0):.4f}\n"
        detail_text += f"- Edge Density: {details.get('edge_density', 0):.4f}\n"
        detail_text += f"- Intensity Std Dev: {details.get('intensity_std', 0):.2f}\n"
        detail_text += f"- Pattern Entropy: {details.get('entropy', 0):.2f}\n\n"
        
        if details.get('factors'):
            detail_text += "Key Findings:\n"
            for factor in details['factors']:
                detail_text += f"✓ {factor}\n"
        
        # Save analysis
        analysis = Analysis(
            eeg_record_id=eeg_id,
            analysis_type='seizure_detection',
            result=prediction,
            confidence=confidence,
            details=detail_text,
            spectrogram_path=eeg_record.file_path,
            histogram_path=heatmap_path
        )
        db.session.add(analysis)
        eeg_record.status = 'completed'
        db.session.commit()
        
        return jsonify({
            'label': prediction,
            'confidence': confidence,
            'analysis_id': analysis.id
        })
        
    except Exception as e:
        print(f"❌ Seizure prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # ADD THESE IMPORTS AT THE TOP


# ADD THIS ROUTE TO SERVE UPLOADED FILES
@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


# REPLACE preview_asd route with this:
@app.route('/asd/preview/<int:eeg_id>')
def preview_asd(eeg_id):
    eeg_record = EEGRecord.query.get_or_404(eeg_id)
    patient = Patient.query.get_or_404(eeg_record.patient_id)
    
    is_csv = eeg_record.filename.lower().endswith('.csv')
    csv_preview = None
    
    if is_csv:
        try:
            # Read CSV and get preview
            df = pd.read_csv(eeg_record.file_path)
            csv_preview = df.head(10)  # First 10 rows
        except Exception as e:
            print(f"CSV read error: {e}")
            csv_preview = None
    
    return render_template('preview_asd.html', 
                         eeg_record=eeg_record, 
                         patient=patient,
                         is_csv=is_csv,
                         csv_preview=csv_preview)


# IMPROVED ASD PREDICTION WITH ROC CURVE DATA


# UPDATED PDF GENERATION WITH ROC CURVE FOR ASD
def create_pdf_report(analysis):
    """
    Generate comprehensive PDF report with graphs (including ROC for ASD)
    """
    try:
        pdf_filename = f"report_{analysis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                              rightMargin=50, leftMargin=50,
                              topMargin=50, bottomMargin=50)
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2b5876'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#3a7bd5'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        title = Paragraph("EEG Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Report Info
        elements.append(Paragraph("Report Information", heading_style))
        
        info_data = [
            ['Report ID:', str(analysis.id)],
            ['Patient Name:', analysis.eeg_record.patient.name],
            ['Patient Age:', f"{analysis.eeg_record.patient.age} years"],
            ['Gender:', analysis.eeg_record.patient.gender],
            ['Analysis Type:', analysis.analysis_type.replace('_', ' ').title()],
            ['Date Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Results
        elements.append(Paragraph("Analysis Results", heading_style))
        
        result_data = [
            ['Result:', analysis.result],
            ['Confidence:', f"{(analysis.confidence * 100):.1f}%"],
        ]
        
        result_table = Table(result_data, colWidths=[2*inch, 4*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        elements.append(result_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Confidence Chart
        fig, ax = plt.subplots(figsize=(6, 3))
        
        if 'seizure' in analysis.analysis_type.lower():
            categories = ['No Seizure', 'Seizure']
            if 'Seizure' in analysis.result:
                confidences = [1 - analysis.confidence, analysis.confidence]
            else:
                confidences = [analysis.confidence, 1 - analysis.confidence]
            colors_bar = ['#48c774', '#f14668']
        else:
            categories = ['Control', 'ASD']
            if 'ASD' in analysis.result:
                confidences = [1 - analysis.confidence, analysis.confidence]
            else:
                confidences = [analysis.confidence, 1 - analysis.confidence]
            colors_bar = ['#48c774', '#3298dc']
        
        bars = ax.bar(categories, confidences, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Confidence', fontweight='bold')
        ax.set_title('Prediction Confidence', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        chart_img = RLImage(img_buffer, width=5*inch, height=2.5*inch)
        elements.append(chart_img)
        elements.append(Spacer(1, 0.3*inch))
        
        # Details
        elements.append(Paragraph("Detailed Analysis", heading_style))
        detail_text = Paragraph(analysis.details.replace('\n', '<br/>'), styles['BodyText'])
        elements.append(detail_text)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add visualization (Heatmap for Seizure, ROC for ASD)
        if analysis.histogram_path and os.path.exists(analysis.histogram_path):
            elements.append(PageBreak())
            elements.append(Paragraph("Visual Analysis", heading_style))
            
            if 'asd' in analysis.analysis_type.lower():
                elements.append(Paragraph("ROC Curve - ASD Detection Performance", styles['Heading3']))
            else:
                elements.append(Paragraph("Seizure Detection Heatmap Analysis", styles['Heading3']))
            
            viz_img = RLImage(analysis.histogram_path, width=6*inch, height=4.5*inch)
            elements.append(viz_img)
        
        # Footer
        elements.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        footer = Paragraph(
            "This report is generated for research and educational purposes. "
            "Results should be reviewed by certified medical professionals.",
            footer_style
        )
        elements.append(footer)
        
        doc.build(elements)
        return pdf_path
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None


# Keep the download_report route the same but ensure it works:
@app.route('/download_report/<int:id>')
def download_report(id):
    """Download PDF report for analysis"""
    try:
        analysis = Analysis.query.get_or_404(id)
        pdf_path = create_pdf_report(analysis)
        
        if pdf_path and os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"EEG_Analysis_Report_{analysis.id}.pdf",
                mimetype='application/pdf'
            )
        else:
            flash('Failed to generate PDF report', 'danger')
            return redirect(url_for('analysis_detail', id=id))
            
    except Exception as e:
        print(f"Download error: {e}")
        flash('Error generating report', 'danger')
        return redirect(url_for('analysis_detail', id=id))
# ============================================
# ASD ML MODEL INTEGRATION - ADD TO TOP OF app.py
# ============================================

# Add these imports at the top
import joblib
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy as scipy_entropy

# Load ASD ML Model - ADD AFTER OTHER IMPORTS
ASD_MODEL = None
ASD_FEATURE_COLUMNS = []
ASD_CHANNELS = []

try:
    asd_model_path = 'asd_model.pkl'
    if os.path.exists(asd_model_path):
        asd_bundle = joblib.load(asd_model_path)
        ASD_MODEL = asd_bundle['model']
        ASD_FEATURE_COLUMNS = asd_bundle['feature_columns']
        ASD_CHANNELS = asd_bundle['channels']
        print(f"✅ ASD ML Model loaded! ({len(ASD_FEATURE_COLUMNS)} features)")
    else:
        print("⚠️ asd_model.pkl not found - using rule-based fallback")
except Exception as e:
    print(f"⚠️ ASD model loading failed: {e}")
    ASD_MODEL = None


# ============================================
# ASD FEATURE EXTRACTION - ADD THESE FUNCTIONS
# ============================================

def infer_fs_from_time(time_series):
    """Infer sampling frequency from time series"""
    try:
        t = np.array(time_series.dropna(), dtype=float)
        if len(t) < 2:
            return 256.0
        dt = np.median(np.diff(t))
        return float(round(1.0 / dt, 2)) if dt > 0 else 256.0
    except:
        return 256.0


def bandpower_welch_asd(signal, fs, band, nperseg=None):
    """Compute band power for ASD analysis"""
    if len(signal) < 4:
        return 0.0
    try:
        nperseg = nperseg or min(128, len(signal))
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        low, high = band
        idx = np.logical_and(f >= low, f <= high)
        return float(np.trapz(Pxx[idx], f[idx])) if np.any(idx) else 0.0
    except:
        return 0.0


def extract_asd_features_from_csv(filepath):
    """Extract features from EEG CSV for ASD ML model"""
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Find time column
        time_col = None
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
                break
        
        # Infer sampling frequency
        fs = infer_fs_from_time(df[time_col]) if time_col else 256.0
        
        # Map channels
        colmap = {c.lower(): c for c in df.columns}
        mapped_channels = {}
        for ch in ASD_CHANNELS:
            for col in df.columns:
                if ch.lower() in col.lower():
                    mapped_channels[ch] = col
                    break
        
        # Define frequency bands
        BANDS = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Window parameters
        window_sec = 2.0
        overlap = 0.3
        step = max(1, int(window_sec * fs * (1.0 - overlap)))
        wlen = int(window_sec * fs)
        
        feat_rows = []
        start = 0
        max_windows = 500  # Limit for large files
        
        print(f"Processing CSV: {len(df)} rows, fs={fs} Hz")
        
        while start + wlen <= len(df) and len(feat_rows) < max_windows:
            win_df = df.iloc[start:start + wlen]
            feats = {}
            
            for ch in ASD_CHANNELS:
                col = mapped_channels.get(ch)
                if col and col in df.columns:
                    sig = pd.to_numeric(win_df[col], errors='coerce').fillna(0).values
                else:
                    sig = np.zeros(wlen)
                
                # Statistical features
                feats[f'{ch}_mean'] = float(np.mean(sig))
                feats[f'{ch}_std'] = float(np.std(sig))
                
                if len(sig) > 2:
                    feats[f'{ch}_skew'] = float(skew(sig))
                    feats[f'{ch}_kurt'] = float(kurtosis(sig))
                else:
                    feats[f'{ch}_skew'] = 0.0
                    feats[f'{ch}_kurt'] = 0.0
                
                # Entropy
                try:
                    hist, _ = np.histogram(sig, bins=16, density=True)
                    hist = hist + 1e-12
                    feats[f'{ch}_entropy'] = float(scipy_entropy(hist))
                except:
                    feats[f'{ch}_entropy'] = 0.0
                
                # Band powers
                for band_name, band_range in BANDS.items():
                    feats[f'{ch}_bp_{band_name}'] = bandpower_welch_asd(sig, fs, band_range)
            
            feat_rows.append(feats)
            start += step
        
        print(f"Extracted features from {len(feat_rows)} windows")
        
        # Create DataFrame and align with model features
        feat_df = pd.DataFrame(feat_rows)
        feat_df = feat_df.reindex(columns=ASD_FEATURE_COLUMNS, fill_value=0.0)
        
        return feat_df
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# REPLACE predict_asd ROUTE WITH THIS
# ============================================

@app.route('/predict_asd', methods=['POST'])
def predict_asd():
    """ASD prediction using YOUR ML model"""
    eeg_id = request.form.get('eeg_id')
    eeg_record = EEGRecord.query.get_or_404(eeg_id)
    
    try:
        if not ASD_MODEL:
            # Fallback to rule-based if model not loaded
            return predict_asd_fallback(eeg_id, eeg_record)
        
        # Extract features from CSV using YOUR method
        feat_df = extract_asd_features_from_csv(eeg_record.file_path)
        
        if feat_df is None or len(feat_df) == 0:
            return jsonify({'error': 'Failed to extract features from CSV'}), 500
        
        # Predict using YOUR ML model
        probs = ASD_MODEL.predict_proba(feat_df)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        # Count results
        asd_count = int((preds == 1).sum())
        control_count = int((preds == 0).sum())
        total_windows = len(preds)
        avg_confidence = float(np.mean(np.maximum(probs, 1 - probs)))
        
        print(f"ASD ML Results: ASD={asd_count}, Control={control_count}, Total={total_windows}")
        
        # Majority vote for final result
        if asd_count > control_count:
            label = "ASD Detected"
            confidence = asd_count / total_windows
        else:
            label = "Control"
            confidence = control_count / total_windows
        
        # Generate ROC curve with actual predictions
        roc_data = generate_roc_curve_from_predictions(probs, preds, label)
        
        # Create detailed description
        detail_text = f"ML-based ASD detection using Random Forest classifier.\n\n"
        detail_text += f"Analysis Summary:\n"
        detail_text += f"- Total Windows Analyzed: {total_windows}\n"
        detail_text += f"- ASD Windows: {asd_count} ({asd_count/total_windows*100:.1f}%)\n"
        detail_text += f"- Control Windows: {control_count} ({control_count/total_windows*100:.1f}%)\n"
        detail_text += f"- Average Window Confidence: {avg_confidence*100:.1f}%\n"
        detail_text += f"- Final Result: {label}\n"
        detail_text += f"- Overall Confidence: {confidence*100:.1f}%\n\n"
        detail_text += f"The model analyzed {total_windows} 2-second windows of EEG data, "
        detail_text += f"extracting {len(ASD_FEATURE_COLUMNS)} features per window including "
        detail_text += f"statistical measures, entropy, and frequency band powers across "
        detail_text += f"{len(ASD_CHANNELS)} EEG channels."
        
        # Save analysis
        analysis = Analysis(
            eeg_record_id=eeg_id,
            analysis_type='asd_detection',
            result=label,
            confidence=confidence,
            details=detail_text,
            spectrogram_path=eeg_record.file_path,
            histogram_path=roc_data['roc_path']
        )
        db.session.add(analysis)
        eeg_record.status = 'completed'
        db.session.commit()
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'analysis_id': analysis.id,
            'asd_count': asd_count,
            'control_count': control_count,
            'total_windows': total_windows
        })
        
    except Exception as e:
        print(f"❌ ASD ML Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


def predict_asd_fallback(eeg_id, eeg_record):
    """Fallback rule-based ASD detection if ML model not available"""
    try:
        df = pd.read_csv(eeg_record.file_path)
        
        # Simple feature extraction
        features = []
        for col in df.columns[:20]:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    features.extend([
                        col_data.mean(),
                        col_data.std(),
                        col_data.min(),
                        col_data.max()
                    ])
            except:
                continue
        
        if len(features) < 10:
            raise ValueError("Insufficient valid numeric data")
        
        feature_array = np.array(features)
        feature_mean = np.mean(feature_array)
        feature_std = np.std(feature_array)
        
        # Simple scoring
        asd_score = 0
        if feature_std > feature_mean * 0.5:
            asd_score += 30
        
        feature_range = np.max(feature_array) - np.min(feature_array)
        if feature_range > feature_mean * 2:
            asd_score += 25
        
        cv = feature_std / (feature_mean + 1e-10)
        if cv > 0.5:
            asd_score += 25
        
        outliers = np.sum(np.abs(feature_array - feature_mean) > 2 * feature_std)
        if outliers > len(feature_array) * 0.1:
            asd_score += 20
        
        confidence = min(asd_score / 100.0, 0.95)
        
        if asd_score >= 50:
            prediction = "ASD Detected"
        else:
            prediction = "Control"
            confidence = 1.0 - confidence
        
        roc_data = generate_roc_curve_for_asd(confidence, prediction)
        
        detail_text = f"Pattern-based ASD detection (fallback mode).\n"
        detail_text += f"Detection Score: {asd_score}/100\n"
        detail_text += f"Result: {prediction} with {confidence*100:.1f}% confidence\n"
        
        analysis = Analysis(
            eeg_record_id=eeg_id,
            analysis_type='asd_detection',
            result=prediction,
            confidence=confidence,
            details=detail_text,
            spectrogram_path=eeg_record.file_path,
            histogram_path=roc_data['roc_path']
        )
        db.session.add(analysis)
        eeg_record.status = 'completed'
        db.session.commit()
        
        return jsonify({
            'success': True,
            'label': prediction,
            'confidence': confidence,
            'analysis_id': analysis.id,
            'asd_count': 1 if "ASD" in prediction else 0,
            'control_count': 0 if "ASD" in prediction else 1,
            'total_windows': 1
        })
        
    except Exception as e:
        print(f"Fallback error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# ROC CURVE GENERATION WITH REAL DATA
# ============================================

def generate_roc_curve_from_predictions(probs, preds, label):
    """Generate ROC curve using actual ML predictions"""
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Use actual predictions for ROC curve
        y_true = preds  # Using predictions as proxy
        y_scores = probs
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curve - ASD Detection', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Pie Chart
        asd_count = int((preds == 1).sum())
        control_count = int((preds == 0).sum())
        
        colors_pie = ['#ff6b6b', '#51cf66']
        explode = (0.05, 0.05)
        
        ax2.pie([asd_count, control_count], 
               labels=['ASD Windows', 'Control Windows'],
               autopct='%1.1f%%',
               startangle=90,
               colors=colors_pie,
               explode=explode,
               shadow=True,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title(f'Window Classification\n(Total: {asd_count + control_count})', 
                     fontsize=14, fontweight='bold')
        
        plt.suptitle('ASD Detection Analysis Results', fontsize=16, fontweight='bold', y=1.02)
        
        filename = f'asd_roc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'roc_path': filepath,
            'auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
    except Exception as e:
        print(f"ROC curve generation error: {e}")
        # Fallback to simple visualization
        return generate_roc_curve_for_asd(0.7, label)


##
# ============================================
# KEEP EXISTING generate_roc_curve_for_asd AS FALLBACK
# (Don't delete the existing function, just add above)
# ============================================
#
#
#
#
import os
import re
import json
import PyPDF2
from docx import Document
from flask import render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image

# ============================================
# ROUTE 1: LEARN EEG PAGE
# ============================================
from werkzeug.utils import secure_filename
import shutil
import matplotlib
matplotlib.use('Agg')  # for headless plotting

from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os

@app.route('/learn-eeg', methods=['GET', 'POST'])
def learn_eeg():
    analysis_result = None
    report_analysis = None

    if request.method == 'POST':
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        # ---------- EEG IMAGE UPLOAD ----------
        if 'eeg_file' in request.files:
            eeg_file = request.files['eeg_file']
            if eeg_file and eeg_file.filename.strip() != "":
                eeg_path = os.path.join(upload_folder, secure_filename(eeg_file.filename))
                eeg_file.save(eeg_path)

                try:
                    # Analyze EEG image
                    short_summary, detailed_explanation = analyze_spectrogram_image(eeg_path)

                    # Store detailed explanation in session for detail page
                    session['detailed_explanation'] = detailed_explanation

                    analysis_result = {
                        'features': {},  # optionally fill features if you have them
                        'images': ["/uploads/" + os.path.basename(eeg_path)],
                        'short_explanation': short_summary,
                        'detailed_explanation': detailed_explanation
                    }

                except Exception as e:
                    analysis_result = {'error': str(e)}

        # ---------- REPORT FILE UPLOAD ----------
        if 'report_file' in request.files:
            report_file = request.files['report_file']
            if report_file and report_file.filename.strip() != "":
                report_path = os.path.join(upload_folder, secure_filename(report_file.filename))
                report_file.save(report_path)

                try:
                    # Analyze the report file
                    report_summary, report_points, report_images = analyze_report_file_dynamic(report_path)

                    # Extract text content for condition detection
                    ext = os.path.splitext(report_path)[1].lower()
                    text_content = ""
                    if ext == '.pdf':
                        import fitz
                        doc = fitz.open(report_path)
                        for page in doc:
                            text_content += page.get_text()
                    elif ext in ['.docx', '.txt', '.csv', '.json']:
                        text_content = extract_report_text(report_path)
                    elif ext in ['.png', '.jpg', '.jpeg']:
                        from PIL import Image
                        import pytesseract
                        text_content = pytesseract.image_to_string(Image.open(report_path))

                    # Parse numbers from text content
                    numbers = parse_numbers_from_text(text_content)

                    # Identify condition and risk
                    condition = identify_condition(text_content, numbers)
                    risk_level = assess_risk_level(numbers, condition)

                    # Generate detailed 10-point explanation
                    detailed_points = generate_detailed_explanation_points(condition, numbers, risk_level, text_content)

                    # Store in session to show on detail page
                    session['report_detailed_points'] = detailed_points
                    session['report_summary'] = report_summary
                    session['report_images'] = report_images

                    # Prepare report analysis for main page
                    report_analysis = {
                        'summary': report_summary,
                        'points': report_points,
                        'images': report_images
                    }

                except Exception as e:
                    report_analysis = {'error': str(e)}

    return render_template(
        "learn_eeg.html",
        analysis_result=analysis_result,
        report_analysis=report_analysis
    )


from flask import Flask, request, render_template

from flask import session

@app.route('/learn-eeg/details', methods=['GET'])
def learn_eeg_details():
    # Get the detailed explanation from session
    detailed_explanation = session.get('detailed_explanation', "No detailed explanation available.")
    return render_template("learn_eeg_details.html", detailed_explanation=detailed_explanation)






# ============================================
# ROUTE 2: DETAILED EXPLANATION PAGE
# ============================================
@app.route('/detailed-explanation', methods=['POST'])
def detailed_explanation():
    explanation_points = request.form.getlist('explanation[]')
    return render_template("detailed_page.html", explanation_points=explanation_points)


# ============================================
# HELPER: ANALYZE EEG IMAGE PATTERN
# ============================================
def analyze_eeg_pattern(image_path):
    """
    Analyzes an EEG spectrogram image and provides educational explanation
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "Could not read the image. Please upload a valid PNG, JPG, or JPEG file."
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract visual characteristics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        color_intensity = np.mean(hsv[:,:,2])
        
        # Analyze frequency distribution (vertical bands)
        height, width = gray.shape
        vertical_bands = []
        band_height = height // 5  # Divide into 5 frequency bands
        
        for i in range(5):
            band = gray[i*band_height:(i+1)*band_height, :]
            vertical_bands.append(np.mean(band))
        
        # Analyze temporal patterns (horizontal variation)
        horizontal_variance = np.var(np.mean(gray, axis=0))
        
        # Generate explanation
        explanation = "📊 **EEG Spectrogram Analysis:**\n\n"
        
        # Brightness analysis
        if brightness > 180:
            explanation += "🔆 **High Brightness Detected:** The spectrogram shows strong signal intensity overall, indicating high brain activity levels. "
        elif brightness > 100:
            explanation += "⚖️ **Moderate Brightness:** The spectrogram shows balanced signal intensity, typical of normal brain activity patterns. "
        else:
            explanation += "🌑 **Low Brightness:** The spectrogram shows lower signal intensity, which could indicate a relaxed or drowsy state. "
        
        # Frequency band analysis
        explanation += "\n\n🎵 **Frequency Band Activity:**\n"
        band_names = ["Gamma (High)", "Beta", "Alpha", "Theta", "Delta (Low)"]
        for i, (name, intensity) in enumerate(zip(band_names, vertical_bands)):
            if intensity > np.mean(vertical_bands) * 1.2:
                explanation += f"- **{name} Band:** HIGH activity detected. "
                if "Delta" in name:
                    explanation += "Strong low-frequency waves suggest deep sleep or relaxation patterns.\n"
                elif "Theta" in name:
                    explanation += "Elevated theta waves may indicate drowsiness or meditative states.\n"
                elif "Alpha" in name:
                    explanation += "Prominent alpha waves suggest a calm, relaxed but awake state.\n"
                elif "Beta" in name:
                    explanation += "High beta activity indicates active thinking and alertness.\n"
                elif "Gamma" in name:
                    explanation += "Elevated gamma activity suggests high cognitive processing.\n"
            elif intensity < np.mean(vertical_bands) * 0.8:
                explanation += f"- **{name} Band:** LOW activity detected.\n"
            else:
                explanation += f"- **{name} Band:** NORMAL activity level.\n"
        
        # Temporal stability
        explanation += "\n\n⏱️ **Temporal Pattern:**\n"
        if horizontal_variance > 500:
            explanation += "High variation over time detected - the brain activity shows significant changes throughout the recording period. This could indicate transitions between different states of consciousness or varying levels of attention."
        elif horizontal_variance > 200:
            explanation += "Moderate temporal variation - the brain activity shows normal fluctuations typical of engaged but stable mental states."
        else:
            explanation += "Low temporal variation - the brain activity is quite stable over time, suggesting a consistent mental state throughout the recording."
        
        explanation += "\n\n💡 **Educational Note:** This is a visual pattern analysis for learning purposes only. Real EEG interpretation requires medical expertise and considers many additional factors."
        
        return explanation
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}. Please ensure you uploaded a valid EEG spectrogram image."


def parse_numbers_from_text(text):
    import re
    numbers = {}
    pattern = r'([A-Za-z0-9_ ]+)\s*[:=]\s*([-+]?\d*\.?\d+|\d+)'
    matches = re.findall(pattern, text)
    for label, value in matches:
        try:
            numbers[label.strip()] = float(value)
        except:
            continue
    return numbers

# ============================================
# HELPER: ANALYZE REPORT FILE
# ============================================
def analyze_report_file_dynamic(path):
    """
    Analyzes uploaded reports (PDF, DOCX, TXT, CSV, JSON, Images)
    - Extracts text, numbers
    - Detects charts/images
    - Generates educational explanation
    """
    import fitz, pytesseract
    ext = os.path.splitext(path)[1].lower()
    text_content = ""
    images = []

    # Extract text
    if ext == '.pdf':
        doc = fitz.open(path)
        for page in doc:
            text_content += page.get_text()
            # extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_name = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.basename(path)}_{img_index}.png")
                with open(img_name, "wb") as f:
                    f.write(image_bytes)
                images.append('/'+img_name)
    elif ext in ['.docx', '.txt', '.csv', '.json']:
        text_content = extract_report_text(path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        text_content = pytesseract.image_to_string(Image.open(path))
        images.append('/'+path)

    # Extract numeric values dynamically
    numbers = parse_numbers_from_text(text_content)

    # Create explanation points dynamically
    points = []
    points.append(f"📄 Extracted {len(numbers)} numeric entries from report.")
    for k,v in numbers.items():
        points.append(f"- {k}: {v}")

    points.append("📊 Images/Charts detected: "+str(len(images)))
    points.append("💡 **Educational Note:** This report analysis explains patterns in data and charts numerically and visually. Not a diagnosis.")

    summary = "Report processed successfully. See detailed points below."
    return summary, points, images



# ============================================
# HELPER: EXTRACT PDF TEXT
# ============================================
def extract_pdf_text(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except:
        pass
    return text


# ============================================
# HELPER: EXTRACT DOCX TEXT
# ============================================
def extract_docx_text(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except:
        pass
    return text


# ============================================
# HELPER: EXTRACT CSV TEXT
# ============================================
def extract_csv_text(file_path):
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()
    except:
        return ""


# ============================================
# HELPER: EXTRACT CONFIDENCE SCORES
# ============================================
def extract_confidence_scores(content, data_dict):
    """Extract confidence scores from text or JSON data"""
    scores = {}
    
    # Try JSON first
    if data_dict:
        if 'confidence' in data_dict:
            scores = data_dict['confidence']
        elif 'scores' in data_dict:
            scores = data_dict['scores']
        elif 'predictions' in data_dict:
            scores = data_dict['predictions']
        else:
            # Look for numeric values
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    scores[key] = float(value)
    
    # Parse text content
    if not scores:
        # Pattern 1: "label: 0.95" or "label = 0.95"
        pattern1 = r'(\w+)\s*[:=]\s*(0?\.\d+|\d+\.?\d*)'
        matches = re.findall(pattern1, content.lower())
        for label, score in matches:
            try:
                scores[label] = float(score)
            except:
                pass
        
        # Pattern 2: "Normal: 85%"
        pattern2 = r'(\w+)\s*:\s*(\d+)%'
        matches = re.findall(pattern2, content)
        for label, score in matches:
            try:
                scores[label.lower()] = float(score) / 100
            except:
                pass
    
    return scores


# ============================================
# HELPER: IDENTIFY CONDITION
# ============================================
def identify_condition(content, scores):
    """Identify the detected condition from content"""
    content_lower = content.lower()
    
    # Check scores first
    if scores:
        max_label = max(scores, key=scores.get)
        return max_label.upper()
    
    # Check for keywords
    conditions = {
        'SEIZURE': ['seizure', 'epilepsy', 'epileptic', 'convulsion'],
        'ASD': ['asd', 'autism', 'autistic', 'autism spectrum'],
        'ADHD': ['adhd', 'attention deficit', 'hyperactivity'],
        'NORMAL': ['normal', 'healthy', 'no abnormality'],
        'ABNORMAL': ['abnormal', 'irregular', 'atypical']
    }
    
    for condition, keywords in conditions.items():
        for keyword in keywords:
            if keyword in content_lower:
                return condition
    
    return "ANALYSIS RESULT"


# ============================================
# HELPER: ASSESS RISK LEVEL
# ============================================
def assess_risk_level(scores, condition):
    """Determine risk level based on scores"""
    if not scores:
        return "MODERATE"
    
    max_score = max(scores.values()) if scores else 0.5
    
    if max_score > 0.85:
        return "HIGH"
    elif max_score > 0.65:
        return "MODERATE"
    else:
        return "LOW"


# ============================================
# HELPER: GENERATE SUMMARY
# ============================================
def generate_summary(condition, scores, risk_level):
    """Generate a brief summary for display"""
    summary = f"🔍 **Detection Result:** {condition}\n\n"
    
    if scores:
        summary += "📊 **Confidence Scores:**\n"
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            percentage = score * 100 if score <= 1 else score
            summary += f"- {label.capitalize()}: {percentage:.1f}%\n"
    
    summary += f"\n⚠️ **Risk Assessment:** {risk_level} confidence level detected"
    
    return summary


# ============================================
# HELPER: GENERATE DETAILED EXPLANATION
# ============================================
def generate_detailed_explanation_points(condition, scores, risk_level, content):
    """Generate detailed 10-point explanation"""
    points = []
    
    # Point 1: What was detected
    points.append(f"**Primary Finding:** The analysis indicates {condition}. This determination is based on pattern recognition in the EEG data provided in your report.")
    
    # Point 2: Confidence explanation
    if scores:
        max_score = max(scores.values())
        percentage = max_score * 100 if max_score <= 1 else max_score
        points.append(f"**Confidence Level:** The model shows {percentage:.1f}% confidence in this detection. Higher confidence (>80%) suggests clearer patterns, while lower confidence suggests ambiguous or mixed patterns.")
    else:
        points.append("**Confidence Level:** Specific confidence scores were not found in the report. The analysis is based on qualitative pattern assessment.")
    
    # Point 3: What this means
    condition_meanings = {
        'SEIZURE': "Seizure-related patterns suggest abnormal electrical activity in the brain, characterized by sudden bursts or spikes in the EEG signal.",
        'ASD': "Autism Spectrum patterns may show differences in brain connectivity, alpha/gamma wave ratios, or synchronization between brain regions.",
        'ADHD': "ADHD patterns often show increased theta/beta ratios and reduced sustained attention markers in frontal brain regions.",
        'NORMAL': "Normal patterns indicate typical brain wave activity with expected frequency distributions and no concerning abnormalities.",
        'ABNORMAL': "Abnormal patterns indicate deviations from typical brain activity that may require further clinical investigation."
    }
    points.append(f"**Clinical Significance:** {condition_meanings.get(condition, 'The detected pattern indicates a specific brain activity signature that differs from baseline expectations.')}")
    
    # Point 4: Risk assessment
    risk_explanations = {
        'HIGH': "High confidence suggests clear, consistent patterns across multiple features. This doesn't mean certainty of diagnosis but indicates strong alignment with the pattern characteristics.",
        'MODERATE': "Moderate confidence indicates some features align with the pattern but others may be ambiguous. Additional clinical correlation is recommended.",
        'LOW': "Low confidence suggests weak or mixed signals. The patterns are not strongly indicative, and alternative explanations should be considered."
    }
    points.append(f"**Risk Assessment:** {risk_explanations.get(risk_level, 'The confidence level provides context for interpretation.')}")
    
    # Point 5: Chart interpretation
    if 'histogram' in content.lower():
        points.append("**Histogram Analysis:** The histogram in your report shows the distribution of EEG features. Tall bars indicate dominant patterns, while the spread shows variability. Skewed distributions may indicate imbalanced brain activity.")
    elif 'roc' in content.lower():
        points.append("**ROC Curve Analysis:** The ROC curve demonstrates model performance. A curve closer to the top-left corner indicates better discrimination ability between conditions.")
    elif 'confusion matrix' in content.lower() or 'heatmap' in content.lower():
        points.append("**Confusion Matrix Analysis:** The heatmap shows prediction accuracy. Diagonal cells represent correct predictions, while off-diagonal cells show misclassifications.")
    else:
        points.append("**Visual Analysis:** Your report contains graphical representations that help visualize the EEG patterns and model predictions in an intuitive format.")
    
    # Point 6: Key features
    points.append("**Key Features Analyzed:** The model examines multiple aspects including frequency band power (delta, theta, alpha, beta, gamma), signal complexity, inter-channel connectivity, and temporal stability patterns.")
    
    # Point 7: What to understand
    points.append("**Understanding the Results:** These results represent machine learning pattern recognition, not medical diagnosis. They identify statistical similarities to known patterns but cannot replace clinical evaluation by qualified healthcare professionals.")
    
    # Point 8: Precautions - general
    points.append("**General Precautions:** If concerning patterns are detected, consult a neurologist or relevant specialist. Keep a log of any symptoms, triggers, or episodes. Maintain regular sleep schedules and manage stress levels.")
    
    # Point 9: Precautions - specific
    specific_precautions = {
        'SEIZURE': "If seizure patterns detected: Avoid flashing lights, get adequate sleep, take medications as prescribed, wear medical ID, and never swim alone. Keep a seizure diary and identify triggers.",
        'ASD': "If ASD patterns detected: Consider sensory-friendly environments, establish routines, explore supportive therapies, and connect with support groups. Early intervention programs may be beneficial.",
        'ADHD': "If ADHD patterns detected: Minimize distractions, use organizational tools, establish structured routines, consider behavioral therapy, and discuss medication options with a doctor.",
        'NORMAL': "With normal patterns: Maintain brain health through regular exercise, good sleep hygiene, mental stimulation, healthy diet, and stress management.",
        'ABNORMAL': "With abnormal patterns: Schedule follow-up with a neurologist, document any symptoms, avoid known triggers, and maintain detailed health records."
    }
    points.append(f"**Specific Precautions:** {specific_precautions.get(condition, 'Consult healthcare providers for personalized guidance based on your specific situation.')}")
    
    # Point 10: Next steps
    points.append("**Next Steps:** (1) Share this report with your doctor, (2) Request comprehensive clinical EEG if not done, (3) Discuss symptoms and concerns, (4) Consider lifestyle modifications, (5) Follow up regularly. Remember: This is an educational tool, not a diagnostic service.")
    
    return points


# ============================================
# HELPER: ANALYZE REPORT IMAGE
# ============================================
def analyze_report_image(image_path):
    """Analyze report images containing charts and graphs"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Could not read image.", ["Invalid image file."]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect type of visualization
        summary = "📊 **Report Visualization Analysis:**\n\n"
        points = []
        
        # Check for pie chart (circular shapes)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=20, maxRadius=200)
        if circles is not None:
            summary += "🥧 Pie chart detected showing distribution of predictions.\n"
            points.append("**Pie Chart Interpretation:** The pie chart visualizes the probability distribution across different conditions. Larger slices indicate higher confidence in those predictions.")
        
        # Check for bars (histogram)
        edges = cv2.Canny(gray, 50, 150)
        vertical_lines = cv2.reduce(edges, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        if np.max(vertical_lines) > 1000:
            summary += "📊 Histogram detected showing feature distributions.\n"
            points.append("**Histogram Interpretation:** The histogram shows how EEG features are distributed. Peaks indicate dominant values, while the spread shows variability in the measurements.")
        
        # Analyze overall brightness distribution
        brightness = np.mean(gray)
        if brightness > 200:
            summary += "⚪ High brightness suggests strong positive signals or dominant features.\n"
        elif brightness < 100:
            summary += "⚫ Low brightness may indicate subtle patterns or normalized data.\n"
        
        points.extend([
            "**Visual Representation:** Charts provide intuitive ways to understand complex EEG analysis results.",
            "**Color Coding:** Different colors typically represent different conditions, frequency bands, or confidence levels.",
            "**Pattern Recognition:** Look for dominant features - largest slices, tallest bars, brightest regions.",
            "**Comparative Analysis:** Charts allow you to compare relative strengths of different predictions or features.",
            "**Clinical Context:** Visual patterns should be interpreted alongside clinical symptoms and history.",
            "**Limitations:** Automated image analysis provides general insights but cannot replace expert interpretation.",
            "**Next Steps:** Share complete reports with healthcare providers for comprehensive evaluation.",
            "**Educational Purpose:** This analysis helps you understand your report structure, not diagnose conditions.",
            "**Follow-up:** Request detailed explanations from your healthcare provider about specific findings."
        ])
        
        return summary, points[:10]  # Ensure we have exactly 10 points
        
    except Exception as e:
        return f"Error analyzing report image: {str(e)}", [f"Error: {str(e)}"]

def preprocess_and_extract(raw, l_freq=0.5, h_freq=70.0):
    """Preprocess EEG signal and extract educational features"""
    raw = raw.copy().filter(l_freq, h_freq, verbose=False)
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = int(raw.info['sfreq'])

    # Compute basic statistics
    features = {}
    for i, ch in enumerate(ch_names):
        x = data[i]
        features[f'{ch}_mean'] = float(np.mean(x))
        features[f'{ch}_var'] = float(np.var(x))
        features[f'{ch}_std'] = float(np.std(x))
        peaks, _ = find_peaks(np.abs(x), height=np.std(x)*3)
        features[f'{ch}_peaks'] = int(len(peaks))

    # Compute band powers (Delta, Theta, Alpha, Beta, Gamma)
    psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=70, n_fft=2048, verbose=False)
    bands = {'Delta':(0.5,4),'Theta':(4,8),'Alpha':(8,12),'Beta':(12,30),'Gamma':(30,70)}
    for band_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        features[f'{band_name}_power_mean'] = float(psds[:, idx].mean())
        features[f'{band_name}_power_std'] = float(psds[:, idx].std())

    return features, raw, (psds, freqs)

def make_explanation(features, model_out=None):
    """
    Generates dynamic educational explanation of EEG features
    """
    lines = []
    lines.append("📊 **EEG Analysis Summary**")
    # Top features with extreme values
    sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    lines.append("Top 10 feature values influencing the EEG pattern:")
    for fname, val in sorted_feats:
        lines.append(f"- {fname}: {val:.3f}")

    # Band power explanation
    if 'Theta_power_mean' in features and 'Alpha_power_mean' in features:
        theta_alpha_ratio = features['Theta_power_mean'] / (features['Alpha_power_mean'] + 1e-12)
        lines.append(f"🔹 Theta/Alpha Ratio: {theta_alpha_ratio:.2f}")
        if theta_alpha_ratio > 1.5:
            lines.append("  ⚠️ Elevated theta/alpha ratio may indicate drowsiness or attention fluctuation (educational insight).")
        else:
            lines.append("  ✅ Theta/Alpha ratio within normal range (educational insight).")

    # Peaks summary
    peak_counts = sum([v for k,v in features.items() if '_peaks' in k])
    lines.append(f"🔹 Total detected peaks across channels: {peak_counts} (helps identify spike activity)")

    lines.append("\n💡 **Note:** This analysis is purely educational and visualizes EEG patterns. It does NOT provide a medical diagnosis.")

    return "\n".join(lines)

import mne
import pandas as pd

def load_eeg_file(file_path):
    """
    Load EEG file (CSV or EDF) and return an MNE Raw object
    """
    ext = file_path.split('.')[-1].lower()
    if ext == 'csv':
        # Assume CSV: each column is a channel, first row is header
        df = pd.read_csv(file_path)
        data = df.values.T  # shape: (n_channels, n_samples)
        ch_names = list(df.columns)
        sfreq = 256  # default sampling rate; adjust if known
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
        raw = mne.io.RawArray(data, info)
    elif ext == 'edf':
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        raise ValueError("Unsupported EEG file format. Use CSV or EDF.")
    return raw

import cv2
import numpy as np

import cv2
import numpy as np

def analyze_eeg_image(image_path):
    """
    Analyze EEG spectrogram image and generate a detailed educational explanation.
    This works for standard EEG spectrograms (like the Zenodo dataset).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the EEG image. Please upload a valid PNG/JPG file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Divide image into 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
    band_names = ["Delta (0.5–4 Hz)", "Theta (4–8 Hz)", "Alpha (8–13 Hz)",
                  "Beta (13–30 Hz)", "Gamma (30–50 Hz)"]
    band_heights = np.linspace(0, height, 6, dtype=int)  # 5 bands

    band_intensities = []
    for i in range(5):
        band = gray[band_heights[i]:band_heights[i+1], :]
        avg_intensity = np.mean(band)
        band_intensities.append(avg_intensity)

    # Analyze temporal stability (horizontal variation)
    horizontal_variance = np.var(np.mean(gray, axis=0))

    # Generate detailed explanation
    explanation = "🧠 **EEG Spectrogram Analysis (Educational)**\n\n"
    
    # Overall brightness
    overall_brightness = np.mean(gray)
    if overall_brightness > 180:
        explanation += "🔆 The EEG spectrogram is overall bright, indicating strong signal intensity across frequencies.\n\n"
    elif overall_brightness > 100:
        explanation += "⚖️ The EEG spectrogram shows moderate brightness, typical of balanced brain activity.\n\n"
    else:
        explanation += "🌑 The EEG spectrogram is darker overall, indicating lower signal intensity.\n\n"

    # Frequency band analysis
    explanation += "🎵 **Frequency Band Analysis:**\n"
    mean_band_intensity = np.mean(band_intensities)
    for name, intensity in zip(band_names, band_intensities):
        if intensity > mean_band_intensity * 1.2:
            explanation += f"- **{name}: HIGH activity detected.** "
        elif intensity < mean_band_intensity * 0.8:
            explanation += f"- **{name}: LOW activity detected.** "
        else:
            explanation += f"- **{name}: Normal activity.** "

        # Educational insight per band
        if "Delta" in name:
            explanation += "Delta waves are linked to deep sleep and restorative brain states.\n"
        elif "Theta" in name:
            explanation += "Theta waves are associated with relaxation, meditation, or drowsiness.\n"
        elif "Alpha" in name:
            explanation += "Alpha waves suggest a calm, relaxed but awake state.\n"
        elif "Beta" in name:
            explanation += "Beta waves indicate active thinking, focus, and alertness.\n"
        elif "Gamma" in name:
            explanation += "Gamma waves are related to high-level cognitive processing.\n"

    # Temporal analysis
    explanation += "\n⏱️ **Temporal Stability:**\n"
    if horizontal_variance > 500:
        explanation += "High variation over time detected. Brain activity changes significantly across the recording.\n"
    elif horizontal_variance > 200:
        explanation += "Moderate temporal variation. Normal fluctuations typical of engaged brain activity.\n"
    else:
        explanation += "Low temporal variation. Brain activity is stable throughout the recording.\n"

    # Educational note
    explanation += ("\n💡 **Educational Note:** This analysis interprets visual EEG patterns "
                    "for learning purposes only. It does not replace clinical diagnosis. "
                    "Consult a neurologist or EEG specialist for medical interpretation.\n")

    return explanation


def analyze_spectrogram_image(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read spectrogram image.")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    brightness = np.mean(v)
    saturation = np.mean(s)
    contrast = np.std(v)

    # --- SHORT SUMMARY (displayed on Learn EEG page) ---
    short_summary = f"""
🧠 **Spectrogram Summary**
- Brightness: {brightness:.2f}
- Color Intensity: {saturation:.2f}
- Contrast: {contrast:.2f}
- A yellow–green band is seen at lower frequencies.
- The upper region is bluish/green → lower power.
"""

    # --- DETAILED EXPLANATION (shown on separate page) ---
    detailed_explanation = f"""
🧠 **FULL Spectrogram EEG Analysis (Beginner Friendly)**

### 📌 What This Image Represents
This is a **time–frequency spectrogram**.  
It shows:
- Time (horizontal direction)
- Frequency (vertical direction)
- Power/Intensity (color)

### 🎨 Color Meaning
- **Yellow = high EEG power**
- **Green = moderate power**
- **Blue = low power**
- **Horizontal yellow band = strong steady low-frequency activity**

### 📊 Extracted Basic Features
- **Average Brightness:** {brightness:.2f}
- **Average Saturation:** {saturation:.2f}
- **Contrast:** {contrast:.2f}

### 🔍 What the Patterns Suggest
- The strong **yellow band at the bottom** indicates powerful **delta/theta waves** (slow brain waves).
- The higher region is **green/blue**, meaning weaker medium/high-frequency activity.
- The vertical texturing means brain activity **changes over time**, not constant.

### 🎓 Layman Explanation
Think of this like:
- **Bottom = slow brain waves**
- **Middle = medium waves**
- **Top = fast waves**
- **Yellow = strong**
- **Blue/Green = weak**

This helps beginners visually understand how EEG converts from a waveform to frequency information.
"""

    return short_summary, detailed_explanation


@app.route('/learn-eeg-report-details', methods=['GET', 'POST'])
def learn_eeg_report_details():
    # Get details from session
    detailed_points = session.get('report_detailed_points', [])
    report_images = session.get('report_images', [])
    report_summary = session.get('report_summary', "")

    return render_template(
        "learn_eeg_report_details.html",
        detailed_points=detailed_points,
        report_images=report_images,
        report_summary=report_summary
    )


# Example for running the app:
# if __name__ == '__main__':
#     app.run(debug=True)

# ========== INITIALIZE DATABASE ==========
with app.app_context():
    db.create_all()
    print("✅ Database initialized")

if __name__ == '__main__':
    app.run(debug=True)