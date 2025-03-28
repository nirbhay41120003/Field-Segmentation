from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
from video_process import process_video
import uuid
from flask import Response
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'input_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Increased to 500MB max file size
app.config['PROCESSED_FOLDER'] = 'processed_video'

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Global variable to track processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{unique_filename}')
        
        # Save the uploaded file
        file.save(input_path)
        
        try:
            # Initialize processing status
            task_id = str(uuid.uuid4())
            processing_status[task_id] = {
                'progress': 0,
                'completed': False,
                'output_path': output_path,
                'filename': f'processed_{unique_filename}'
            }

            # Process the video
            process_video(
                input_video_path=input_path,
                output_video_path=output_path,
                model_path=r"C:\Users\nirbh\OneDrive\Desktop\det\last.pt",
                conf_threshold=0.2
            )
            
            # Update status to completed
            processing_status[task_id]['completed'] = True
            processing_status[task_id]['progress'] = 100
            
            # Return task ID and filename
            return jsonify({
                'success': True,
                'task_id': task_id,
                'filename': f'processed_{unique_filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up input file
            if os.path.exists(input_path):
                os.remove(input_path)
                
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    return jsonify({'error': 'Task not found'}), 404

@app.route('/video/<filename>')
def get_video(filename):
    try:
        video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(video_path):
            return send_file(
                video_path,
                mimetype='video/mp4',
                as_attachment=False
            )
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_video(filename):
    try:
        return send_file(
            os.path.join(app.config['PROCESSED_FOLDER'], filename),
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
