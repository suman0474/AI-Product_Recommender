from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGTEXT
db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
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

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # User who owns the project
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Project basic info
    project_name = db.Column(db.String(200), nullable=False)
    project_description = db.Column(db.Text, nullable=True)
    
    # Original requirements from project page
    initial_requirements = db.Column(db.Text, nullable=False)
    
    # Product type identified
    product_type = db.Column(db.String(100), nullable=True)
    
    # Identified instruments and accessories from project page
    identified_instruments = db.Column(db.Text, nullable=True)  # JSON string
    identified_accessories = db.Column(db.Text, nullable=True)  # JSON string
    
    # All collected data through conversation
    collected_data = db.Column(db.Text, nullable=True)  # JSON string
    
    # All chat messages/conversation logs
    conversation_history = db.Column(db.Text, nullable=True)  # JSON string
    
    # Search tabs that were opened
    search_tabs = db.Column(db.Text, nullable=True)  # JSON string
    
    # Final analysis results if completed
    analysis_results = db.Column(db.Text, nullable=True)  # JSON string
    
    # Project state information
    current_step = db.Column(db.String(50), nullable=True)
    project_status = db.Column(db.String(20), default='active', nullable=False)  # active, completed, archived
    
    # Timestamps
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = db.relationship('User', backref=db.backref('projects', lazy=True))
