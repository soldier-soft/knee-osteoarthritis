from flask import Blueprint, render_template, request, current_app, flash, redirect, url_for, Response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import csv
import io
from services.ml_service import predict_label
from services.db_service import insert_prediction, get_user_predictions, get_prediction_by_id, delete_prediction
from utils.image_utils import validate_image
from config import Config

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route("/")
@main_bp.route("/first")
def first():
    return render_template('first.html')

@main_bp.route("/prevention")
def prevention():
    return render_template('prevention.html')

@main_bp.route("/chart")
@login_required
def chart():
    predictions = get_user_predictions(current_user.id)
    
    # Initialize all 5 levels explicitly requested
    severity_counts = {
        "Normal": 0, "Doubtful": 0, "Mild": 0, "Moderate": 0, "Severe": 0
    }
    
    timeline_data = []
    
    for p in predictions:
        res = p.get('result', 'Normal')
        if res in severity_counts:
            severity_counts[res] += 1
            
        # Parse timestamp safely
        tstamp = p.get('timestamp')
        if tstamp:
            date_str = tstamp.strftime('%Y-%m-%d %H:%M')
            timeline_data.append({
                "date": date_str,
                "confidence": float(p.get('confidence', 0)),
                "severity": res
            })

    # Sort timeline chronological for line/scatter plot
    timeline_data.reverse()

    return render_template('chart.html', 
        severity_counts=severity_counts, 
        timeline_data=timeline_data,
        total_preds=len(predictions)
    )

@main_bp.route("/performance")
def performance():
    return render_template('performance.html')

@main_bp.route("/index", methods=['GET', 'POST'])
@login_required
def index():
    return render_template("index.html")

@main_bp.route("/dashboard")
@login_required
def dashboard():
    predictions = get_user_predictions(current_user.id)
    return render_template('dashboard.html', predictions=predictions)

@main_bp.route("/view/<prediction_id>")
@login_required
def view_prediction(prediction_id):
    prediction = get_prediction_by_id(current_user.id, prediction_id)
    if not prediction:
        flash('Prediction not found.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    filename = prediction['image_path'].split('\\')[-1] if '\\' in prediction['image_path'] else prediction['image_path'].split('/')[-1]
    
    return render_template(
        "result.html", 
        prediction=prediction['result'], 
        confidence=prediction['confidence'], 
        is_blurry=prediction.get('is_blurry', False),
        filename=filename,
        img_path=f"static/{prediction['image_path']}"
    )

@main_bp.route("/delete/<prediction_id>", methods=['POST'])
@login_required
def delete_prediction_route(prediction_id):
    prediction = get_prediction_by_id(current_user.id, prediction_id)
    if prediction:
        delete_prediction(current_user.id, prediction_id)
        flash('Prediction record deleted successfully.', 'success')
    else:
        flash('Prediction not found.', 'danger')
    return redirect(url_for('main.dashboard'))

@main_bp.route("/export_csv")
@login_required
def export_csv():
    predictions = get_user_predictions(current_user.id)
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['X-Ray Image Path', 'Upload Date', 'Diagnosis', 'Confidence', 'Low Quality'])
    for p in predictions:
        cw.writerow([
            p.get('image_path', ''),
            p.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if p.get('timestamp') else 'N/A',
            p.get('result', 'Unknown'),
            f"{p.get('confidence', 0)}%",
            'Yes' if p.get('is_blurry') else 'No'
        ])
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=prediction_history.csv"}
    )

@main_bp.route("/submit", methods=['POST'])
@login_required
def get_output():
    if 'my_image' not in request.files:
        flash('No file provided.', 'danger')
        return redirect(url_for('main.index'))
        
    img = request.files['my_image']
    if img.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('main.index'))

    if not allowed_file(img.filename):
        flash('Warning: Invalid file format. Please upload a valid image file (PNG, JPG, or JPEG).', 'danger')
        return redirect(url_for('main.index'))

    filename = secure_filename(img.filename)
    upload_folder = Config.UPLOAD_FOLDER
    os.makedirs(upload_folder, exist_ok=True)
    img_path_full = os.path.join(upload_folder, filename)
    img_path_rel = f"tests/{filename}"
    
    img.save(img_path_full)

    import time
    # Intentional processing delay (~10s) to simulate deep medical analysis
    # This allows the frontend JS to cycle through its detailed progress loading messages
    time.sleep(10)

    # 4-Level Strict Image Validation Pipeline
    report = validate_image(img_path_full)
    
    if not report['is_valid']:
        flash(report['message'], 'warning')
        os.remove(img_path_full) 
        return redirect(url_for('main.index'))
        
    if report['is_blurry']:
        flash(report['message'], 'warning')

    try:
        # Final Classification + Grad-CAM Heatmap
        predict_result, confidence, heatmap_path_full = predict_label(img_path_full)
        
        # Strict Output Validation Check
        if confidence < 85.0:
            flash(f"Uncertain result ({confidence}%). Artificial Intelligence confidence is too low. Please upload a higher quality, clearer image.", 'warning')
            os.remove(img_path_full)
            if os.path.exists(heatmap_path_full) and heatmap_path_full != img_path_full:
                os.remove(heatmap_path_full)
            return redirect(url_for('main.index'))
            
        heatmap_filename = os.path.basename(heatmap_path_full)
        heatmap_path_rel = f"tests/{heatmap_filename}"
        
        # Store basic stats in DB
        insert_prediction(
            user_id=current_user.id,
            image_path=img_path_rel,
            result=predict_result,
            confidence=confidence,
            is_blurry=report['is_blurry']
        )

        return render_template(
            "result.html", 
            prediction=predict_result, 
            confidence=confidence, 
            is_blurry=report['is_blurry'],
            filename=filename,
            img_path=f"static/{img_path_rel}",
            heatmap_path=f"static/{heatmap_path_rel}",
            report=report
        )
                               
    except Exception as e:
        flash(f"Error processing image: {str(e)}", "danger")
        return redirect(url_for('main.index'))
