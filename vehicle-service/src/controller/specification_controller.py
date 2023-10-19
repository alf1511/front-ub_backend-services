class SpecificationController:
    def __init__(self, specification):
        self.specification_model = specification 

    def process_set_specification(self, qr):
        self.specification_model.set_specification(qr)

    def judgements(self, result, save_results=False):
        if result:
            self.specification_model.update_specification(result)
        
        for k, v in result.items():
            part = k.split('-')[0].strip()
            if part in self.specification_model.spec:
                self.specification_model.spec[part][3].append('{:.3f}%'.format(v))
    
        if save_results:
            self.specification_model.save_vehicle()
            self.specification_model.save_specification()