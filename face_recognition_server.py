import os
import time
import json
import numpy as np
import face_recognition
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
import pickle
from threading import Lock
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access if needed

# Configuration
UPLOAD_FOLDER = 'uploads'
FACES_FOLDER = os.path.join(UPLOAD_FOLDER, 'faces')
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')
ENCODINGS_FILE = 'face_encodings.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FACE_MATCH_TOLERANCE = 0.6  # Lower is more strict

# Create directories if they don’t exist
os.makedirs(FACES_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configure logging
handler = RotatingFileHandler('face_recognition_server.log', maxBytes=100000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Add console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
app.logger.addHandler(console_handler)

# Thread safety for facial encodings
encodings_lock = Lock()
student_encodings = {}

# Load existing encodings if available
def load_encodings():
    global student_encodings
    with encodings_lock:
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    student_encodings = pickle.load(f)
                app.logger.info(f"Loaded {len(student_encodings)} face encodings")
            except Exception as e:
                app.logger.error(f"Error loading encodings: {e}")
                student_encodings = {}
        else:
            app.logger.info("No existing encodings file found, starting with empty dataset")
            student_encodings = {}

# Save encodings to disk
def save_encodings():
    with encodings_lock:
        try:
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump(student_encodings, f)
            app.logger.info(f"Saved {len(student_encodings)} face encodings")
        except Exception as e:
            app.logger.error(f"Error saving encodings: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract the face encoding from an image
def extract_face_encoding(image_path):
    try:
        # Load image
        app.logger.info(f"Processing image: {image_path}")
        if not os.path.exists(image_path):
            app.logger.error(f"Image file does not exist: {image_path}")
            return None
            
        image = face_recognition.load_image_file(image_path)
        app.logger.info(f"Image loaded, shape: {image.shape}")
        
        # Find face locations (using HOG-based model for speed)
        face_locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=1)
        
        if not face_locations:
            app.logger.warning(f"No faces found in {image_path}")
            return None
        
        if len(face_locations) > 1:
            app.logger.warning(f"Multiple faces found in {image_path}, using the first one")
        
        # Get face encodings for the first face found
        face_encodings = face_recognition.face_encodings(
            image, 
            face_locations, 
            num_jitters=1  # More jitters = more accuracy but slower
        )
        
        if not face_encodings:
            app.logger.warning(f"Could not compute encoding for face in {image_path}")
            return None
            
        # Return the first face encoding
        app.logger.info(f"Successfully extracted face encoding")
        return face_encodings[0]
        
    except Exception as e:
        app.logger.error(f"Error extracting face encoding: {e}")
        app.logger.error(traceback.format_exc())
        return None

@app.route('/register', methods=['POST'])
def register_face():
    try:
        app.logger.info("Received registration request")
        
        if 'image' not in request.files:
            app.logger.warning("No image file provided in registration request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            app.logger.warning("Empty filename in registration request")
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            app.logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400
        
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        
        app.logger.info(f"Processing registration for student: {student_name} (ID: {student_id})")
        
        if not student_id:
            app.logger.warning("Student ID is required but not provided")
            return jsonify({"error": "Student ID is required"}), 400
            
        # Save the image
        filename = secure_filename(f"{student_id}_{int(time.time())}.jpg")
        image_path = os.path.join(FACES_FOLDER, filename)
        file.save(image_path)
        app.logger.info(f"Saved image to {image_path}")
        
        # Extract face encoding
        face_encoding = extract_face_encoding(image_path)
        
        if face_encoding is None:
            # Clean up the file if processing failed
            if os.path.exists(image_path):
                os.remove(image_path)
                app.logger.info(f"Removed image file after failed encoding: {image_path}")
            return jsonify({"error": "No valid face found in the image"}), 400
        
        # Store the encoding
        with encodings_lock:
            if student_id in student_encodings:
                student_encodings[student_id]["encodings"].append(face_encoding)
                app.logger.info(f"Added new encoding for existing student {student_id}")
            else:
                student_encodings[student_id] = {
                    "name": student_name,
                    "image_path": image_path,
                    "encodings": [face_encoding]
                }
                app.logger.info(f"Created new student entry with encoding for {student_id}")
        
        # Save updated encodings
        save_encodings()
        
        return jsonify({
            "success": True,
            "student_id": student_id,
            "message": "Face registered successfully"
        }), 201
    except Exception as e:
        app.logger.error(f"Error during face registration: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        app.logger.info("Received recognition request")
        
        if 'image' not in request.files:
            app.logger.warning("No image file provided in recognition request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            app.logger.warning("Empty filename in recognition request")
            return jsonify({"error": "No file selected"}), 400
            
        if not allowed_file(file.filename):
            app.logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save the temp image
        filename = secure_filename(f"temp_{int(time.time())}.jpg")
        image_path = os.path.join(TEMP_FOLDER, filename)
        file.save(image_path)
        app.logger.info(f"Saved image to {image_path}")
        
        try:
            # Extract face encoding
            face_encoding = extract_face_encoding(image_path)
            
            if face_encoding is None:
                # Clean up the temp file
                if os.path.exists(image_path):
                    os.remove(image_path)
                return jsonify({"error": "No valid face found in the image"}), 400
            
            # Check if we have any encodings to match against
            with encodings_lock:
                if not student_encodings:
                    app.logger.warning("No student encodings available to match against")
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    return jsonify({
                        "success": True,
                        "recognized": False,
                        "message": "No registered faces to match against"
                    })
            
            # Match against stored encodings
            best_match = None
            best_distance = 1.0  # Lower is better
            
            with encodings_lock:
                app.logger.info(f"Matching against {len(student_encodings)} student records")
                for student_id, data in student_encodings.items():
                    app.logger.info(f"Checking student {student_id} ({data['name']}) with {len(data['encodings'])} encodings")
                    for stored_encoding in data["encodings"]:
                        # Calculate face distance
                        face_distances = face_recognition.face_distance([stored_encoding], face_encoding)
                        distance = face_distances[0]
                        app.logger.info(f"Distance to {student_id}: {distance}")
                        
                        # Check if this is a better match
                        if distance < FACE_MATCH_TOLERANCE and distance < best_distance:
                            best_distance = distance
                            best_match = {
                                "student_id": student_id,
                                "name": data["name"],
                                "confidence": float(1 - distance)  # Convert distance to confidence score
                            }
            
            # Clean up the temp file
            if os.path.exists(image_path):
                os.remove(image_path)
                app.logger.info(f"Removed temporary image: {image_path}")
            
            if best_match:
                app.logger.info(f"Face recognized as {best_match['name']} with confidence {best_match['confidence']}")
                return jsonify({
                    "success": True,
                    "recognized": True,
                    "student": best_match,
                    "confidence": best_match["confidence"]
                })
            else:
                app.logger.info("No matching face found")
                return jsonify({
                    "success": True,
                    "recognized": False,
                    "message": "No matching face found"
                })
                
        finally:  # Ensure cleanup even if an error occurs
            if os.path.exists(image_path):
                os.remove(image_path)
                app.logger.info(f"Removed temporary image after processing: {image_path}")
            
    except Exception as e:
        app.logger.error(f"Error in recognize endpoint: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/students', methods=['GET'])
def get_students():
    try:
        with encodings_lock:
            students = [
                {
                    "id": student_id,
                    "name": data["name"],
                    "face_count": len(data["encodings"])
                }
                for student_id, data in student_encodings.items()
            ]
        
        app.logger.info(f"Returning {len(students)} students")
        return jsonify({"students": students})
    except Exception as e:
        app.logger.error(f"Error getting students: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/delete/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        app.logger.info(f"Received request to delete student {student_id}")
        with encodings_lock:
            if student_id in student_encodings:
                # Get the student data
                student = student_encodings[student_id]
                
                # Try to remove the image file
                try:
                    if os.path.exists(student["image_path"]):
                        os.remove(student["image_path"])
                        app.logger.info(f"Removed image file: {student['image_path']}")
                except Exception as e:
                    app.logger.error(f"Error removing image file: {e}")
                
                # Remove the student from encodings
                del student_encodings[student_id]
                app.logger.info(f"Deleted student {student_id} from encodings")
                
                # Save updated encodings
                save_encodings()
                
                return jsonify({"success": True, "message": "Student deleted successfully"})
            else:
                app.logger.warning(f"Student {student_id} not found")
                return jsonify({"error": "Student not found"}), 404
    except Exception as e:
        app.logger.error(f"Error deleting student: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    try:
        with encodings_lock:
            status_data = {
                "status": "online",
                "registered_students": len(student_encodings),
                "total_faces": sum(len(data["encodings"]) for data in student_encodings.values()),
                "face_match_tolerance": FACE_MATCH_TOLERANCE
            }
        
        app.logger.info(f"Status request: {status_data}")
        return jsonify(status_data)
    except Exception as e:
        app.logger.error(f"Error getting status: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "error": str(e)}), 500

# Serve uploaded face images if needed
@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Load existing encodings on startup
    load_encodings()
    
    # Log startup information
    app.logger.info("=== Face Recognition Server Starting ===")
    app.logger.info(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    app.logger.info(f"FACES_FOLDER: {FACES_FOLDER}")
    app.logger.info(f"TEMP_FOLDER: {TEMP_FOLDER}")
    app.logger.info(f"FACE_MATCH_TOLERANCE: {FACE_MATCH_TOLERANCE}")
    
    # Start the server
    app.run(host='0.0.0.0', port=5005, debug=False)