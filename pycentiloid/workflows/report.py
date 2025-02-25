from pathlib import Path
from typing import Dict, Optional
from ..reporting.report_generator import AmyloidReportGenerator
from ..centiloid.calculation import CentiloidCalculator
from ..utils.aal import calculate_aal_suvr

class AmyloidReportWorkflow:
    def __init__(self, tracer_type: str):
        self.tracer_type = tracer_type
        self.report_generator = AmyloidReportGenerator()
        self.centiloid_calculator = CentiloidCalculator(tracer_type)
    
    def generate_report(self,
                       pet_path: str,
                       subject_info: Dict,
                       output_dir: Optional[Path] = None) -> str:
        """
        Generate comprehensive Amyloid PET report.
        """
        # 1. Calculate Centiloid score
        centiloid_score = self.centiloid_calculator.calculate(pet_path)
        
        # 2. Calculate regional SUVr values
        aal_suvr = calculate_aal_suvr(pet_path)
        
        # 3. Generate report
        report_path = self.report_generator.generate_report(
            pet_data=pet_path,
            centiloid_score=centiloid_score,
            aal_suvr_values=aal_suvr,
            output_dir=output_dir,
            subject_info=subject_info
        )
        
        return report_path