from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGTEXT
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    # User's full name
    first_name = db.Column(db.String(80), nullable=True)
    last_name = db.Column(db.String(80), nullable=True)
    # NEW: Add a status column for admin approval
    status = db.Column(db.String(20), default='pending', nullable=False)
    # Optional: Add a role column to differentiate admins from regular users
    role = db.Column(db.String(20), default='user', nullable=False)

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # Use ForeignKey to link to the User who performed the action
    user_name = db.Column(db.String(80), nullable=False)
    
    # Store the initial user query (can be long, so use Text)
    user_query = db.Column(db.Text, nullable=False)
    
    # Store the JSON system response as a string
    system_response = db.Column(db.Text, nullable=False)
    
    # Store the feedback text
    feedback = db.Column(db.String(255), nullable=True)
    
    # Automatically set the timestamp when a log is created
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class StandardsDocument(db.Model):
    id = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    content_type = db.Column(db.String(255), nullable=True)
    file_type = db.Column(db.String(32), nullable=True)
    raw_blob_path = db.Column(db.Text, nullable=True)
    extracted_blob_path = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(32), nullable=False, default='uploaded')
    character_count = db.Column(db.Integer, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)