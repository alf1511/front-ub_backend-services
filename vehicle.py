import json
import datetime
import os

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
print(os.listdir('./'))
with open('config.json') as config_file:
    data = json.load(config_file)
    model_config = data["vehicle_name"]

class DB_Vehicle(db.Model) :
    __tablename__ = 'vehicles'
    id = db.Column(db.Integer, primary_key=True)
    nvcss_code = db.Column(db.String(100), unique=False, nullable=False)
    model_no = db.Column(db.String(100), nullable=False)
    frame_no = db.Column(db.String(100), nullable=False)
    suffix = db.Column(db.String(100), nullable=False)
    # timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now().strftime("%y-%m-%d") )
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())
    specifications = db.relationship('DB_Specification', backref='vehicle', lazy=True)

    def __init__(self, nvcss_code):
        self.nvcss_code = nvcss_code
        self.model_no = self.get_model()
        self.frame_no = self.get_frame_no()
        self.suffix = self.get_suffix()
    
    def get_vehicle_name(self):
        code = int(self.model_no)
        model = model_config.get(str(code), 'NOT FOUND')
        return model
    
    def get_suffix(self):
        return self.nvcss_code[19:21]
    
    def get_frame_no(self):
        return self.nvcss_code[21:38]
    
    def get_model(self):
        return self.nvcss_code[5:7] 

class DB_Specification(db.Model) :
    __tablename__ = 'specifications_judgement'
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'))
    partname = db.Column(db.String(100), nullable=False)
    actual_specification = db.Column(db.String(100), nullable=False)
    detected_specification = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(100), nullable=False)

    def __init__(self, vehicle, partname, actual_specification, detected_specification, confidence, status):
        self.vehicle_id = vehicle.id
        self.partname = partname
        self.actual_specification = actual_specification
        self.detected_specification = detected_specification
        self.confidence = confidence
        self.status = status