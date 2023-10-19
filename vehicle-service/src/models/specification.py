import pandas as pd

from .database.vehicle import db, DB_Vehicle, DB_Specification

class Specification:
    def __init__(self, data, is_database = False):
        self.qr = ""
        self.variant = ""
        self.suffix_no = ""
        self.frame_no = ""
        self.spec = {}
        self.vehicle_standard = {}
        self.vehicle_actual = {}
        self.status = {}
        self.vehicle_name = data
        self.is_database = is_database
        if (self.is_database):
            self.vehicle_model = DB_Vehicle(self.qr)

    def compare(self, a, b):
        a = a.lower()
        b = b.lower()
        return [c for c in a if c.isalpha()] == [c for c in b if c.isalpha()]
    
    def equality_check(self, arr1_ori, arr2_ori, size1, size2):
        if (size1 != size2):
            return False
        arr1 = arr1_ori.copy()
        arr2 = arr2_ori.copy()
        arr1.sort()
        arr2.sort()
        for i in range(0, size2):
            arr1[i] = arr1[i].lower()
            arr2[i] = arr2[i].lower()
            if (arr1[i] != arr2[i]):
                return False
        return True

    def set_specification(self, qr):
        self.qr = qr
        self.vehicle_model = DB_Vehicle(self.qr)
        self.variant = self.vehicle_model.get_vehicle_name()
        self.suffix_no = self.vehicle_model.suffix
        self.frame_no = self.vehicle_model.frame_no
        self.spec = {}
        self.load_specification()

    def load_specification(self):
        print(self.variant)
        url = 'suffix_data/Suffix_{}.xlsx'.format(self.variant)
        vehicle_df = pd.read_excel(url, engine='openpyxl')
        
        try:
            vehicle_df[self.suffix_no]
        except:
            self.suffix_no = int(self.suffix_no)

        for i in range(len(vehicle_df[self.suffix_no])):
            self.spec[vehicle_df["PARTNAME"][i]] = [[],[],[],[],[]]
            
            split_ampersand = vehicle_df[self.suffix_no][i].split("&")
            split_semicolon = vehicle_df[self.suffix_no][i].split(";")

            if len(split_ampersand) > 1:
                for item in split_ampersand:
                    self.spec[vehicle_df["PARTNAME"][i]][0].append(item.strip())
                    self.spec[vehicle_df["PARTNAME"][i]][2]='WAIT'
            elif len(split_semicolon) > 1:
                for item in split_semicolon:
                    self.spec[vehicle_df["PARTNAME"][i]][0].append(item.strip())
                    self.spec[vehicle_df["PARTNAME"][i]][2]='WAIT'
            else:
                self.spec[vehicle_df["PARTNAME"][i]][0].append(vehicle_df[self.suffix_no][i].strip())
                self.spec[vehicle_df["PARTNAME"][i]][2]='WAIT'

    def update_specification(self, parts):
        for part in parts:
            part = part.split("-")
            part[0] = part[0].strip()
            if len(part) > 1:
                part[1] = part[1].strip()
                for key in self.spec:
                    if self.compare(key, part[0]) and part[1] not in self.spec[key][1]:
                        self.spec[key][1].append(part[1])
                        if self.equality_check(self.spec[key][0], self.spec[key][1], len(self.spec[key][0]), len(self.spec[key][1])):
                            self.spec[key][2] = 'OK'
                        else:
                            self.spec[key][2] = 'NG'
    
    def get_specification(self):
        return self.spec

    # Save vehicle data to database
    def save_vehicle(self) :
        if self.is_database :
            print("saving vehicle")
            try :
                db.session.add(self.vehicle_model)
                db.session.flush()
                db.session.expunge(self.vehicle_model)
                db.session.commit() 
                print("Vehicle saved to database")
            except Exception as e:
                print('Failed to insert vehicle: ', str(e))
        else:
            print("Vehicle not saved to database, check your db connection or set database to True in confir.json")

    # Save specification data to database
    def save_specification(self):
        if self.is_database:
            print("saving specification")
            print(self.spec)
            for key in self.spec:
                partname = key
                standard_spec = self.spec[key][0]
                detected_spec = self.spec[key][1]
                confidence = self.spec[key][3]
                status = self.spec[key][2]
                db.session.flush()
                specification_model = DB_Specification(self.vehicle_model, partname, standard_spec, detected_spec, confidence, status)
                try:
                    db.session.add(specification_model)  # Add the CarSpecLog instance to the session
                    db.session.commit()  # Save the changes to the database
                    print("Specification saved to database")
                except Exception as e:
                    db.session.rollback()  # Rollback the changes in case of an exception
                    print("Failed to insert specification: ", str(e))
        else:
            print("Specification not saved to database, check your db connection or set database to True in confir.json")
